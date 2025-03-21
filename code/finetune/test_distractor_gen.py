# Reference: https://github.com/umass-ml4ed/distractor-ranking/blob/main/disgen_dpo.py

import wandb
import time
import pandas as pd
import hydra
from tqdm import tqdm
import json
from transformers import AutoTokenizer

from code.utils.utils import set_random_seed, sanitize_configs, save_csv, get_device, clean_distractor
from code.finetune.batch_collator import CollateWrapperGenerativeTest
from code.utils.data_utils import get_stop_token_info
from code.utils.load_data import load_data_finetune, get_test_data_loader_finetune
from code.distractorgen.test import compute_metrics, beam_search
from code.finetune.model import LanguageModel


def test_distractor_gen(test_set, wandb_run_name, configs, device):
    if( configs.task_name == "d_given_s_e" ):
        assert configs.num_distractor_samples == 1, "Top one distractor per question-error pair, set num_distractor_samples to 1"
        gt_distractors = [question['option'][dis_idx] for question in test_set for dis_idx in range(3)]
        # Normalize by number of unique qid/err pairs
        # This ensures that matching *any* distractor (up to out of 3) from the set of a unique qid/err pair records a single match for that pair
        num_errors = sum([len(set(question["misconception_name"])) for question in test_set])

    configs.testing = True
    # Load model 
    model = LanguageModel(configs, device, "test", wandb_run_name).to(device)
    # Load tokenizer
    tokenizer_dir = f"{configs.model_checkpoint_dir}/{configs.task_name}/{wandb_run_name}/best_val_loss/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"

    if( configs.task_name == "d_given_s_e" ):
        df_test = pd.DataFrame(test_set)
        df_test = df_test.explode(["option", "option_idx", "explanation", "misconception_id", "misconception_name"])
        test_set = json.loads(df_test.to_json(orient="records"))
    # Get test data loader
    test_loader = get_test_data_loader_finetune(test_set, CollateWrapperGenerativeTest, tokenizer, device, configs)

    # Generate distractors
    start_time = time.time()
    distractors = []
    cots = []
    stop_token, stop_token_id = get_stop_token_info(configs.task_name, tokenizer)
    # Increase number of distractor samples for e_d_given_s task since different e can map to same d causing duplicates
    num_distractor_samples = configs.num_distractor_samples * 10 if configs.task_name == "e_d_given_s" else configs.num_distractor_samples
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("[Distractor Gen] Batch {}".format(batch_num))
            distractors_batch, _, cots_batch = beam_search(batch, model, tokenizer, stop_token, stop_token_id, configs, num_distractor_samples, configs.num_beams, "distractor_gen")
            distractors = distractors + distractors_batch
            cots = cots + cots_batch

    if( configs.task_name == "d_given_s_e" ):
        distractors = [dis[0] for dis in distractors]
        cots = [""] * len(distractors)

    # Add padding distractors in case there aren't k unique per question after dropping duplicates, add idx so not dropped as duplicates
    if( configs.task_name == "e_d_given_s" ):
        pad_distractors = [f"(empty{idx})" for idx in range(configs.num_distractor_samples)]
        pad_cots = [f"" for _ in range(configs.num_distractor_samples)]
        for _, (row_distractors, row_cots) in enumerate(zip(distractors, cots)):
            row_distractors.extend(pad_distractors)
            row_cots.extend(pad_cots)

    df_test = pd.DataFrame(test_set)
    df_test["predicted_distractor"] = distractors

    if( configs.task_name == "e_d_given_s" ):
        cots_col_nm = "predicted_error"
    else:
        cots_col_nm = "predicted_chain_of_thought"
    df_test[cots_col_nm] = cots

    df_test = df_test.explode(["predicted_distractor", cots_col_nm])

    # Distractor post processing: remove duplicates from pool of 100, keep first 10 unique distractors
    if( configs.task_name == "e_d_given_s" ):
        # Distractors are return sorted from best to worst according to beam score
        df_test = df_test.drop_duplicates(subset=["qid", "predicted_distractor"], keep="first")
        df_test = df_test.groupby("qid").head(configs.num_distractor_samples).reset_index(drop=True)
        # Assert that there are k unique distractors per question
        assert df_test.groupby("qid").size().min() == configs.num_distractor_samples
    
    test_time = time.time() - start_time
    if( configs.log_wandb ):
        wandb.log({"logs/test/time": test_time})
    print(f"logs/test/time: {test_time}s")
    
    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    if( configs.task_name in ["d_given_s_e", "e_d_given_s"] ):
        # Save distractors individually with errors
        filename = f"baseline_{configs.task_name.replace('_', '-')}_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_num-beams-{configs.num_beams}_seed-{configs.seed}_{model_name}_{wandb_run_name}_with_e"
        filename = filename + "_dbg" if configs.debug else filename
        save_csv(df_test, filename, results_dir)
        
        # Record metrics
        filename = f"baseline_{configs.task_name.replace('_', '-')}_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_num-beams-{configs.num_beams}_seed-{configs.seed}_{model_name}_{wandb_run_name}"
        filename = filename + "_dbg" if configs.debug else filename
        if( configs.task_name == "d_given_s_e" ):
            pred_distractors = distractors
            matched = [clean_distractor(gt_dis) == clean_distractor(pred_dis) for gt_dis, pred_dis in zip(gt_distractors, pred_distractors)]
            print(f"Matched: {sum(matched) / num_errors}")
            with open(f"{results_dir}/{filename}.txt", "w") as f:
                f.write(f"Matched: {sum(matched) / num_errors}")
            if( configs.log_wandb ):
                wandb.log({"logs/test/matched": sum(matched) / num_errors})
        else:
            df_test = compute_metrics(df_test, configs, filename)
            # Save grouped distractors without errors
            save_csv(df_test, filename, results_dir)


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    # Get device
    device = get_device(configs)
    # Load data
    mode = "without_error_label" if configs.task_name == "d_given_s" else "with_error_label"
    _, _, test_set = load_data_finetune(configs, mode)

    test_distractor_gen(test_set, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    main()