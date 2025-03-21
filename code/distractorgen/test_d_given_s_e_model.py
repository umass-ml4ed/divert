"""
Test p(d|s,e) from vae post vae training.
"""

import hydra
import pandas as pd
from tqdm import tqdm
import json
from transformers import AutoTokenizer, logging

from code.utils.utils import get_device, set_random_seed, sanitize_configs, clean_distractor, save_csv
from code.utils.load_data import load_data_distractorgen, get_test_data_loader_distractorgen
from code.distractorgen.batch_collator import CollateWrapperGenerativeTestSampleDistractorsGivenQuestionError
from code.distractorgen.model import DistractorGenModel
from code.utils.data_utils import get_stop_token_info
from code.distractorgen.test import beam_search


def test(test_set, wandb_run_name, configs, device):
    gt_distractors = [question['option'][dis_idx] for question in test_set for dis_idx in range(3)]
    # Normalize by number of unique qid/err pairs
    # This ensures that matching *any* distractor (up to out of 3) from the set of a unique qid/err pair records a single match for that pair
    num_errors = sum([len(set(question["misconception_name"])) for question in test_set])

    configs.testing = True
    # Load model
    pipeline = DistractorGenModel(configs, device, "test", wandb_run_name).to(device)
    # Load tokenizer
    tokenizer_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/d_given_s_e/best_val_loss/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"

    df_test = pd.DataFrame(test_set)
    df_test = df_test.explode(["option", "option_idx", "explanation", "misconception_id", "misconception_name"])
    test_set = json.loads(df_test.to_json(orient="records"))

    # Get test data loader
    test_loader = get_test_data_loader_distractorgen(test_set, CollateWrapperGenerativeTestSampleDistractorsGivenQuestionError, tokenizer, device, configs, pipeline, configs.test_batch_size)

    distractors = []
    stop_token, stop_token_id = get_stop_token_info("d_given_s_e", tokenizer)
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("[Distractor Gen] Batch {}".format(batch_num))
            assert configs.num_distractor_samples == 1, "Top one distractor per question-error pair, set num_distractor_samples to 1"
            distractors_batch, _, _ = beam_search(batch, pipeline, tokenizer, stop_token, stop_token_id, configs, configs.num_distractor_samples, configs.num_beams_distractor, "distractor_gen", "d_given_s_e_adapter")
            distractors = distractors + distractors_batch
    pred_distractors = [dis[0] for dis in distractors]
    df_test = pd.DataFrame(test_set)
    df_test["predicted_distractor"] = pred_distractors

    matched = [clean_distractor(gt_dis) == clean_distractor(pred_dis) for gt_dis, pred_dis in zip(gt_distractors, pred_distractors)]
    print(f"Matched: {sum(matched) / num_errors}")
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_seed-{configs.seed}_{model_name}_{wandb_run_name}_d_given_s_e_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    with open(f"{results_dir}/{filename}.txt", "w") as f:
        f.write(f"Matched: {sum(matched) / num_errors}")

    # Save predictions
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-d-{configs.num_distractor_samples}_seed-{configs.seed}_{model_name}_{wandb_run_name}_d_given_s_e"
    filename = filename + "_dbg" if configs.debug else filename
    save_csv(df_test, filename, results_dir)


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
    _, _, _, _, test_set_with_err = load_data_distractorgen(configs)

    test(test_set_with_err, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    main()