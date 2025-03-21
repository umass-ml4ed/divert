import time
import wandb
import pandas as pd
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import torch

from code.utils.utils import save_csv, get_device, set_random_seed, sanitize_configs
from code.utils.load_data import load_data_finetune, get_test_data_loader_finetune
from code.utils.data_utils import get_stop_token_info
from code.finetune.batch_collator import CollateWrapperGenerativeTest
from code.finetune.model import LanguageModel
from code.distractorgen.eval_errors import eval_errors
from code.distractorgen.test import beam_search


def test_error_gen(test_set, wandb_run_name, configs, device):
    configs.testing = True
    # Load model 
    model = LanguageModel(configs, device, "test", wandb_run_name).to(device)

    # Load tokenizer
    if( configs.zero_shot ):
        tokenizer_dir = configs.model_name
    else:
        tokenizer_dir = f"{configs.model_checkpoint_dir}/{configs.task_name}/{wandb_run_name}/best_val_loss/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"

    if( configs.task_name == "e_given_s_d" ):
        df_test = pd.DataFrame(test_set)
        df_test = df_test.explode(["option", "option_idx", "explanation", "misconception_id", "misconception_name", "proportion"])
        test_set = json.loads(df_test.to_json(orient="records"))
    # Get test data loader
    test_loader = get_test_data_loader_finetune(test_set, CollateWrapperGenerativeTest, tokenizer, device, configs)

    # Run batched inference
    start_time = time.time()
    errors = []
    stop_token, stop_token_id = get_stop_token_info(configs.task_name, tokenizer)
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("[Error Gen] Batch {}".format(batch_num))
            # Beam search (optionally with diversity)
            errors_batch, _, _ = beam_search(batch, model, tokenizer, stop_token, stop_token_id, configs, configs.num_error_samples, configs.num_beams, "error_gen")
            errors = errors + errors_batch
    if( configs.task_name == "e_given_s_d" ):
        # List of single error -> single error, will be pooled to list grouped by qid below
        errors = [err[0] for err in errors]
    df_test = pd.DataFrame(test_set)
    df_test["predicted_error"] = errors
    if( configs.task_name == "e_given_s_d" ):
        # Pool back
        g = df_test.groupby("qid")
        list_cols = ["predicted_error", "option", "option_idx", "explanation", "misconception_id", "misconception_name", "proportion"]
        df_test = g.agg({col: lambda x: list(x) for col in list_cols}).join(g[[col for col in df_test.columns if col not in ["qid"]+list_cols]].nth(0)).reset_index()
    df_test = df_test.explode("predicted_error")
    test_time = time.time() - start_time

    # Log metrics to weights and biases
    if( configs.log_wandb ):
        wandb.log({"logs/test/time": test_time})
    print(f"logs/test/time: {test_time}s")

    # Compute metrics
    model_name = configs.model_name.split('/')[1].lower() if len(configs.model_name.split('/')) > 1 else configs.model_name.split('/')[0].lower()
    filename = f"baseline_{configs.task_name.replace('_', '-')}_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}_e_metrics"
    filename = filename + "_dbg" if configs.debug else filename
    mode = "match_one" if configs.task_name == "e_given_s_d" else "match_any"
    permute = False if configs.task_name == "e_given_s_d" else True
    eval_errors(df_test, "predicted_error", configs, filename, mode, permute)

    # Save predictions
    filename = f"baseline_{configs.task_name.replace('_', '-')}_f{configs.cross_val_fold}_tgen-{configs.topic_generalization}_num-e-{configs.num_error_samples}_divbeam-e-{configs.diverse_beam_search_error_gen}_seed-{configs.seed}_{model_name}_{wandb_run_name}"
    filename = filename + "_dbg" if configs.debug else filename
    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    save_csv(df_test, filename, results_dir)


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    configs.log_wandb = False
    # Get device
    device = get_device(configs)
    # Load data
    _, _, test_set = load_data_finetune(configs, mode="with_error_label")

    test_error_gen(test_set, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    main()