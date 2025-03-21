"""
Test p(e|s) from vae post vae training. Errors generated are not linked to end distractors unlike full vae pipeline test.
"""

import hydra
import pandas as pd
from transformers import AutoTokenizer, logging
import sys

from code.utils.utils import get_device, set_random_seed, sanitize_configs, save_csv
from code.utils.load_data import load_data_distractorgen, get_test_data_loader_distractorgen
from code.distractorgen.batch_collator import CollateWrapperGenerativeTestSampleErrors
from code.distractorgen.model import DistractorGenModel
from code.distractorgen.eval_errors import eval_errors
from code.distractorgen.test import generate_errors_e_given_s_vae


def test(test_set, wandb_run_name, configs, device):
    configs.testing = True
    # Load model
    pipeline = DistractorGenModel(configs, device, "test", wandb_run_name).to(device)
    # Load tokenizer
    tokenizer_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/e_given_s/best_val_loss/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"
    # Get test data loader to generate errors using p(e|s)
    test_loader = get_test_data_loader_distractorgen(test_set, CollateWrapperGenerativeTestSampleErrors, tokenizer, device, configs, pipeline, configs.test_batch_size)

    # Generate errors
    errors, _ = generate_errors_e_given_s_vae(test_loader, tokenizer, pipeline, configs, device)
    # Add sampled errors from p(e|s) to test set
    df_test = pd.DataFrame(test_set)
    df_test["predicted_error_p_e_given_s"] = errors
    df_test = df_test.explode(["predicted_error_p_e_given_s"])

    # Compute metrics
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_given_s_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    eval_errors(df_test, "predicted_error_p_e_given_s", configs, filename)

    # Save predictions
    filename = f"vae_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_given_s"
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
    _, _, _, test_set, _ = load_data_distractorgen(configs)

    test(test_set, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    main()