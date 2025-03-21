import hydra
import wandb
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import json
import os
from transformers import AutoTokenizer, logging
from transformers.generation.beam_search import BeamSearchScorer

from code.utils.utils import get_device, set_random_seed, sanitize_configs, save_csv, clean_distractor, find_distractor_cot
from code.utils.load_data import load_data_distractorgen, get_test_data_loader_distractorgen, load_df
from code.distractorgen.batch_collator import (
    CollateWrapperGenerativeTestSampleErrors, CollateWrapperGenerativeTestSampleDistractors)
from code.distractorgen.model import DistractorGenModel
from code.utils.data_utils import get_stop_token_info, post_process_predictions
from code.distractorgen.eval_errors import eval_errors
from code.utils.beam_score_patch import patch_beam_scorer


def generate_errors_e_given_s_vae(test_loader, tokenizer, pipeline, configs, device):
    errors = []
    sample_scores = []
    stop_token, stop_token_id = get_stop_token_info("e_given_s", tokenizer)
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("[Error Gen] Batch {}".format(batch_num))
            errors_batch, scores_batch, _ = beam_search(batch, pipeline, tokenizer, stop_token, stop_token_id, configs, configs.num_error_samples, configs.num_beams_error, "error_gen", "e_given_s_adapter")
            errors = errors + errors_batch
            sample_scores = sample_scores + scores_batch
    errors = [list(error) for error in errors]

    return (errors, sample_scores)


def test(test_set, wandb_run_name, configs, device):
    configs.testing = True

    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_num-e-{configs.num_error_samples}_rnk-{configs.rank}_divbeam-e-{configs.diverse_beam_search_error_gen}_base-{configs.use_base_models}_seed-{configs.seed}_{model_name}_{wandb_run_name}_with_e"
    filename = filename + "_dbg" if configs.debug else filename

    dis_file_exists = os.path.exists(os.path.join(results_dir, filename + "_dis.csv"))

    replace_file = True
    if replace_file or not os.path.exists(os.path.join(results_dir, filename + ".csv")):
        # Load model
        pipeline = DistractorGenModel(configs, device, "test", wandb_run_name, configs.use_base_models).to(device)
        # Load tokenizer
        tokenizer_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/d_given_s_e/best_val_loss/lora_model"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # Batched inference requires tokenizer padding side as left
        tokenizer.padding_side = "left"

        start_time = time.time()

        if not dis_file_exists:
            # Get test data loader to generate errors using p(e|s)
            test_loader = get_test_data_loader_distractorgen(test_set, CollateWrapperGenerativeTestSampleErrors, tokenizer, device, configs, pipeline, configs.test_batch_size)

            # Generate errors
            errors, sample_scores = generate_errors_e_given_s_vae(test_loader, tokenizer, pipeline, configs, device)
            # Add sampled errors from p(e|s) to test set
            df_test = pd.DataFrame(test_set)
            df_test["predicted_error_p_e_given_s"] = errors
            df_test["error_score"] = sample_scores
            df_test = df_test.explode(["predicted_error_p_e_given_s", "error_score"])

            # Generate distractors
            test_set = json.loads(df_test.to_json(orient="records"))
            # Get test data loader to generate distractors using p(d|s,e)
            test_batch_size = configs.test_batch_size
            test_loader = get_test_data_loader_distractorgen(test_set, CollateWrapperGenerativeTestSampleDistractors, tokenizer, device, configs, pipeline, test_batch_size)

            distractors = []
            beam_scores = []
            stop_token, stop_token_id = get_stop_token_info("d_given_s_e", tokenizer)
            with tqdm(test_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("[Distractor Gen] Batch {}".format(batch_num))
                    # Generate three unique distractors per question-error pair using beam search, three unique distractors per question-error pair ensures one unique distractor per error per question after processing
                    distractors_batch, beam_scores_batch, cots_batch = beam_search(batch, pipeline, tokenizer, stop_token, stop_token_id, configs, configs.num_distractor_samples, configs.num_beams_distractor, "distractor_gen", "d_given_s_e_adapter")
                    beam_scores = beam_scores + beam_scores_batch
                    distractors = distractors + distractors_batch
            df_test = pd.DataFrame(test_set)

        # Select distractors from p(d|s,e) and add to test set
        if( configs.rank ):
            if not dis_file_exists:
                # Add padding distractors in case there aren't k unique per question after cleaning, add idx so not dropped as duplicates
                for idx, (row_distractors, row_scores) in enumerate(zip(distractors, beam_scores)):
                    row_distractors.append(f"(empty{idx})")
                    row_scores.append(-np.inf)
                # Rank overgenerated distractors using beam scores and error scores.
                # Doesn't ensure one unique distractor per error per question.
                df_test["predicted_distractor"] = distractors
                df_test["distractor_beam_score"] = beam_scores
                df_test = df_test.explode(["predicted_distractor", "distractor_beam_score"])
                save_csv(df_test, filename + "_dis", results_dir)
            else:
                df_test = load_df(filename + "_dis.csv", results_dir,
                                    other_obj_cols=["option", "option_idx", "explanation", "misconception_id", "misconception_name", "proportion"])
            map_at_k = configs.map_at_k if configs.map_at_k != 0 else configs.num_distractor_samples
            df_test = rank_distractors(df_test, map_at_k, configs.rank_with_distractor_scores_only)
        else:
            #assert configs.num_distractor_samples == 3, "Number of distractor samples should be 3 for processing distractor beams"
            # Pool distractors ensuring one unique distractor per error per question
            distractors = process_distractor_beams(distractors, configs)
            df_test["predicted_distractor"] = distractors

        # Save distractors individually with errors
        save_csv(df_test, filename, results_dir)

        test_time = time.time() - start_time
        if( configs.log_wandb ):
            wandb.log({"logs/test/time": test_time})
        else:
            print(f"logs/test/time: {test_time}s")

    else:
        print("Result file already exists - skipping generation")
        df_test = load_df(filename + ".csv", results_dir, other_obj_cols=["option"])

    # Evaluate errors from p(e|s)
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_num-e-{configs.num_error_samples}_rnk-{configs.rank}_divbeam-e-{configs.diverse_beam_search_error_gen}_base-{configs.use_base_models}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    eval_errors(df_test, "predicted_error_p_e_given_s", configs, filename)

    # Record metrics
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_num-e-{configs.num_error_samples}_rnk-{configs.rank}_divbeam-e-{configs.diverse_beam_search_error_gen}_base-{configs.use_base_models}_seed-{configs.seed}_{model_name}_{wandb_run_name}_d_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    df_test = compute_metrics(df_test, configs, filename)

    # Save grouped distractors without errors
    save_csv(df_test, filename, results_dir)


def rank_distractors(df: pd.DataFrame, k: int, rank_with_distractor_scores_only: bool):
    if( rank_with_distractor_scores_only ):
        df["score"] = df["distractor_beam_score"]
    else:
        df["score"] = df["distractor_beam_score"] + df["error_score"]
    # Sort by ["qid", "score"], then drop duplicate distractors keeping the first occurence with highest score 
    df = df.sort_values(["qid", "score"], ascending=False).drop_duplicates(subset=["qid", "predicted_distractor"], keep="first")
    df = df.groupby("qid").head(k).reset_index(drop=True)

    return df


def process_distractor_beams(distractors, configs):
    k = configs.num_error_samples
    top_unique_distractors = []
    # Pick top unique distractor for each question-error pair from three unique distractors per question-error pair ensuring one unique distractor per error per question
    for i in range(0, len(distractors), k):
        distractor_lists = distractors[i:i+k]
        top_unique_distractors_per_question = pick_top_unique_distractors_per_question(distractor_lists)
        top_unique_distractors += list(top_unique_distractors_per_question)

    return top_unique_distractors


def pick_top_unique_distractors_per_question(distractor_lists):
    top_unique_distractors_per_question = []    
    for l in distractor_lists:
        flag_found = False
        for distractor in l:
            if( distractor != "" and distractor not in top_unique_distractors_per_question ):
                top_unique_distractors_per_question.append(distractor)
                flag_found = True
                break
        if( not flag_found ):
            top_unique_distractors_per_question.append("")

    return top_unique_distractors_per_question


def beam_search(batch, pipeline, tokenizer, stop_token, stop_token_id, configs, num_unique_samples, num_beams, mode="distractor_gen", adapter_name=None):
    patch_beam_scorer(BeamSearchScorer) # Patch scorer functionality to include prompt length in penalty
    beam_scores = None 
    cots_batch = None
    diversity_params = {}

    if( mode == "error_gen" ):
        max_new_tokens = configs.max_new_tokens_error
        if( configs.diverse_beam_search_error_gen ):
            diversity_params = {"num_beam_groups" : num_unique_samples, "diversity_penalty" : configs.diversity_penalty}
    else:
        # Mode is distractor generation
        if( configs.exp_name == "pretrain" and configs.task_name == "e_d_given_s" ):
            max_new_tokens = configs.max_new_tokens_distractor + configs.max_new_tokens_error
        else:
            max_new_tokens = configs.max_new_tokens_distractor
    # Repeat prompts_len num_unique_samples times for each question
    prompts_len = [[x]*num_unique_samples for x in batch["prompts_len"]]
    prompts_len = [x for l in prompts_len for x in l]
    if( adapter_name != None ):
        with pipeline.model_ctm(adapter_name, "eval"): 
            outputs = pipeline.model.generate(
                input_ids=batch["inputs"]["input_ids"],
                attention_mask=batch["inputs"]["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_unique_samples,
                do_sample=False,
                num_beams=num_beams,
                eos_token_id = stop_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                length_penalty = configs.length_penalty,
                **diversity_params
            )

    else:
        outputs = pipeline.model.generate(
            input_ids=batch["inputs"]["input_ids"],
            attention_mask=batch["inputs"]["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_unique_samples,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id = stop_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty = configs.length_penalty,
            **diversity_params
        )

    beam_scores = outputs.sequences_scores.reshape(-1, num_unique_samples).tolist()

    preds_batch = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    preds_batch = post_process_predictions(preds_batch, prompts_len, stop_token, configs)
    if( mode == "distractor_gen" ):
        cots = []
        distractors = []
        for pred in preds_batch:
            distractor, cot = find_distractor_cot(pred, configs)
            distractors.append(clean_distractor(distractor))
            cots.append(cot)
        preds_batch = distractors
        cots_batch = [cots[i:i+num_unique_samples] for i in range(0, len(cots), num_unique_samples)]
    results_batch = [preds_batch[i:i+num_unique_samples] for i in range(0, len(preds_batch), num_unique_samples)]

    return (results_batch, beam_scores, cots_batch)


def count_match(pred_distractors, ground_truth_distractors, k):
    # Use sets to avoid counting duplicates twice
    pred_distractors = set(pred_distractors[:k])
    ground_truth_distractors = set(ground_truth_distractors)
    num_matched = min(3, len(pred_distractors.intersection(ground_truth_distractors)))

    return num_matched


def compute_metrics(df_test, configs, filename):
    print(f"--- Evaluating distractors ---")
    # Pool predicted distractors for each question
    g = df_test.groupby("qid")
    df_test = g.agg({"predicted_distractor": lambda x: list(x)}).join(g[["option", "question", "lvl_2_subject_id", "lvl_2_subject_name", "lvl_3_subject_id", "lvl_3_subject_name", "construct_id", "construct_name", "correct_answer", "solution"]].nth(0)).reset_index()
    
    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    with open(f"{results_dir}/{filename}.txt", "w") as f:
        map_at_k = configs.map_at_k if configs.map_at_k != 0 else configs.num_distractor_samples
        for k in range(1, map_at_k + 1):
            df_test[f"count_match_at_{k}"] = df_test.apply(lambda row: count_match(row["predicted_distractor"], row["option"], k), axis=1)
            df_test[f"exact_match_at_{k}"] = (df_test[f"count_match_at_{k}"] == 3).astype(int)
            df_test[f"partial_match_at_{k}"] = (df_test[f"count_match_at_{k}"] > 0).astype(int)

            exact_match = 100 * df_test[f"exact_match_at_{k}"].sum() / len(df_test)
            partial_match = 100 * df_test[f"partial_match_at_{k}"].sum() / len(df_test)
            proportional_match = 100 * df_test[f"count_match_at_{k}"].sum() / (3 * len(df_test))

            f.write(f"map@{k}\n")
            f.write(f"metrics/test/exact_at_{k}: {exact_match}%\n")
            f.write(f"metrics/test/partial_at_{k}: {partial_match}%\n")
            f.write(f"metrics/test/proportional_at_{k}: {proportional_match}%\n")
            print(f"map@{k}")
            print(f"metrics/test/exact_at_{k}: {exact_match}%")
            print(f"metrics/test/partial_at_{k}: {partial_match}%")
            print(f"metrics/test/proportional_at_{k}: {proportional_match}%")
            if( configs.log_wandb ):
                wandb.log({f"metrics/test/exact_at_{k}": exact_match})
                wandb.log({f"metrics/test/partial_at_{k}": partial_match})
                wandb.log({f"metrics/test/proportional_at_{k}": proportional_match})
    print(f"--- Done evaluating distractors ---")

    return df_test


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Turn off warnings
    logging.set_verbosity_error()
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    configs.log_wandb = False # Will crash if set to true since no associated run
    # Get device
    device = get_device(configs)
    # Load data
    _, _, _, test_set, _ = load_data_distractorgen(configs)

    test(test_set, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    main()