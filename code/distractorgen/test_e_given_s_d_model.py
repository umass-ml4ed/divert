"""
Test q(e|s,d) from vae post vae training.
"""

import hydra
import pandas as pd
from tqdm import tqdm
import time
import json
from transformers import AutoTokenizer, logging

from code.utils.utils import get_device, set_random_seed, sanitize_configs, save_csv
from code.utils.load_data import load_data_distractorgen, get_test_data_loader_distractorgen
from code.distractorgen.batch_collator import CollateWrapperGenerativeTestSampleErrorsGivenQuestionDistractor
from code.distractorgen.model import DistractorGenModel
from code.utils.data_utils import get_stop_token_info
from code.distractorgen.eval_errors import eval_errors
from code.distractorgen.test import beam_search


def test(test_set, wandb_run_name, configs, device):
    configs.testing = True
    # Load model
    pipeline = DistractorGenModel(configs, device, "test", wandb_run_name).to(device)
    # Load tokenizer
    tokenizer_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/e_given_s_d/best_val_loss/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"

    df_test = pd.DataFrame(test_set)
    df_test = df_test.explode(["option", "option_idx", "explanation", "misconception_id", "misconception_name"])
    test_set = json.loads(df_test.to_json(orient="records"))

    # Get test data loader to generate errors using p(e|s,d)
    test_loader = get_test_data_loader_distractorgen(test_set, CollateWrapperGenerativeTestSampleErrorsGivenQuestionDistractor, tokenizer, device, configs, pipeline, configs.test_batch_size)

    # Run batched inference
    start_time = time.time()
    errors = []
    stop_token, stop_token_id = get_stop_token_info("e_given_s_d", tokenizer)
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("[Error Gen] Batch {}".format(batch_num))
            # Beam search (with diversity)
            errors_batch, _, _ = beam_search(batch, pipeline, tokenizer, stop_token, stop_token_id, configs, configs.num_error_samples, configs.num_beams_error, "error_gen", "e_given_s_d_adapter")
            errors = errors + errors_batch
    print(f"errors: {errors}")
    df_test = pd.DataFrame(test_set)
    df_test["predicted_error"] = errors
    df_test = df_test.explode("predicted_error")
    test_time = time.time() - start_time

    # Compute metrics
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_given_s_d_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    eval_errors(df_test, "predicted_error", configs, filename, mode="match_one", permute=False)

    # Save predictions
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_given_s_d"
    filename = filename + "_dbg" if configs.debug else filename
    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
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