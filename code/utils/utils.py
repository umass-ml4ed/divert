import torch
import random
import pathlib
import re
import os
import pandas as pd
import numpy as np
import gc
from rouge_score import rouge_scorer
from itertools import chain
import wandb
from pprint import pprint


PRED_RE = re.compile(r"(?s)(.*?)The incorrect answer given by the student is:(.*)")


def find_distractor_cot(pred, configs):
    cot = ""
    distractor = ""
    if( configs.exp_name == "pretrain" and configs.task_name == "e_d_given_s" ):
        match = PRED_RE.search(pred)
        if match:
            cot = match.group(1)
            distractor = match.group(2)
    else:
        distractor = pred

    return (distractor, cot)


def is_batch_left(iter_data_loader):
    return len(iter_data_loader) - iter_data_loader._num_yielded > 0


def num_batches_left(iter_data_loader):
    return len(iter_data_loader) - iter_data_loader._num_yielded


def clean_distractor(string):
    string = string.lower()

    # Standardize symbols
    string = string.replace("\\%", "%")
    string = string.replace("...", "\\ldots")
    string = string.replace('÷', '\\div')
    string = string.replace('≥', '\\geq')
    string = string.replace('≤', '\\leq')
    string = string.replace('≠', '\\neq')
    string = string.replace('≈', '\\approx')
    string = string.replace('δ', '\\delta')
    string = string.replace('|', '\\vert')

    # Remove math environment indicators
    string = string.replace("$", "")
    string = string.replace("\\[", "")
    string = string.replace("\\]", "")
    string = string.replace("\\(", "")
    string = string.replace("\\)", "")

    # convert / and \div fractions to \frac
    string = re.sub(r"([\d\.]+)\s*(/|\\div)\s*([\d\.]+)", r"\\frac{\g<1>}{\g<3>}", string) 
    # convert x to \times
    string = re.sub(r'\s*×\s*', r' \\times ', string)
    # convert √ to \\sqrt{}
    string = re.sub(r'√', r'\\sqrt', string) 
    # convert 2 cm to 2 \mathrm{~cm}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*cm',  r'\1 \\mathrm{~cm}', string)
    # convert 2 m to 2 \mathrm{~m}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*m',  r'\1 \\mathrm{~m}', string)
    # convert 2 km to 2 mathrm{~km}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*km',  r'\1 \\mathrm{~km}', string)

    # convert p^2 to p^{2}
    string = re.sub(r'([a-zA-Z])\^(\d+)', r'\1^{\2}', string)

    # remove hyphen between words
    string = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1\2', string)

    string = string.replace('\\mathrm{~m}athrm{~cm}', '\\mathrm{~cm}')
    string = string.replace('\\mathrm{~m}ore', 'more')
    string = string.replace(' ', '')
    string = string.strip()

    return string


def save_sampled_errors(batches, cur_iter, configs, wandb_run_name):
    data = {}
    for key in batches[0].keys():
        data[key] = list(chain.from_iterable([batch[key] for batch in batches]))
    df = pd.DataFrame.from_dict(data)
    checkpoint_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/sampled_errors"
    save_csv(df, f"epoch_{cur_iter}_sampled_errors", checkpoint_dir)


def get_adapter_dirs(configs):
    if( configs.model_name == "gpt2" ):
        adapter_dirs = {
            "e_given_s_d_adapter": "model_dir_e_given_s_d_gpt2",
            "e_given_s_adapter": "model_dir_e_given_s_gpt2",
            "d_given_s_e_adapter": "model_dir_d_given_s_e_gpt2",
            "e_given_s_d_ref_adapter": "model_dir_e_given_s_d_gpt2"
        }
    elif( configs.model_name == "meta-math/MetaMath-Mistral-7B" ):
        adapter_dirs = {
            "e_given_s_d_adapter": "model_dir_e_given_s_d_metamath_mistral",
            "e_given_s_adapter": "model_dir_e_given_s_metamath_mistral",
            "d_given_s_e_adapter": "model_dir_d_given_s_e_metamath_mistral",
            "e_given_s_d_ref_adapter": "model_dir_e_given_s_d_metamath_mistral"
        }

    for k, v in adapter_dirs.items():
        adapter_dirs[k] = os.path.join(configs.pretrain_model_dir_root, configs[v], configs.pretrain_model_dir_suffix)
    print("Adapter directories:")
    pprint(adapter_dirs)

    return adapter_dirs


def get_gpu_memory():
    print("-".join(["-" for _ in range(5)]))
    print("GPU memory status:")
    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024.0/1024/1024}GB")
    print(f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024.0/1024/1024}GB")
    print(f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024.0/1024/1024}GB")
    print("-".join(["-" for _ in range(5)]))


def remove_full_stop(text):
    text = text.strip()
    if( text[-1] == "." ):
        text = text[:-1]

    return text


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_question(text):
    # Replace whitespace except newline with a single space
    text = re.sub(r"[^\S\r\n]+", " ", text).strip()
    # Replace multiple newlines with a single newline since we use double newline as a delimiter in our prompt
    text = re.sub(r"\n+", "\n", text).strip()

    return text


def clean_str(text):
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_str_punct_end(text):
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text).strip()
    # Punctuate end of sentence if missing punctuation
    if text[-1] not in [".", "!", "?"]:
        text += "."
    
    return text


def save_model(trainer, configs, wandb_run_name, name="best_val_loss"):
    if( configs.exp_name == "distractorgen" ):
        for adapter_name in trainer.pipeline.adapter_names:
            checkpoint_dir = f"{configs.model_checkpoint_dir}/{wandb_run_name}/{adapter_name.split('_adapter')[0]}/{name}/lora_model"
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            # Save tokenizer
            trainer.tokenizer.save_pretrained(checkpoint_dir)
            # Save LoRA model only, not complete model
            trainer.pipeline.model.save_pretrained(checkpoint_dir, selected_adapters=[f"{adapter_name}"])
    elif( configs.exp_name == "pretrain" ):
        checkpoint_dir = f"{configs.model_checkpoint_dir}/{configs.task_name}/{wandb_run_name}/best_val_loss/lora_model"
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Save tokenizer
        trainer.tokenizer.save_pretrained(checkpoint_dir)
        # Save LoRA model only, not complete model
        trainer.model.model.save_pretrained(checkpoint_dir)
    else:
        raise Exception("Error: Invalid configs.exp_name")


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    else:
        return x.detach().cpu().numpy()


def aggregate_metrics(outputs):
    res = {}
    for k in outputs[0].keys():
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        res[k] = np.mean(all_logs)

    return res


def sanitize_configs(configs):
    if( configs.testing ):
        configs.log_wandb = False
    if( configs.debug ):
        configs.num_epochs = 1
        configs.log_wandb = False
    if( configs.exp_name == "distractorgen" ):
        assert configs.beta_kl >= 0 and configs.beta_kl <= 1, "Error: Value of beta_kl not in [0, 1]"

    return configs


def compute_rouge_l_f1(targets, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for (target, prediction) in zip(targets, predictions):
        scores.append(scorer.score(target, prediction)["rougeL"].fmeasure)
    rouge_l_f1 = np.mean(np.asarray(scores, dtype=np.float32))

    return rouge_l_f1


def save_csv(df, filename, dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)


def get_device(configs):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if configs.use_cuda: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        assert device.type == 'cuda', 'Error: No GPU found'
    else:
        device = torch.device('cpu')

    return device


def run_garbage_collector():
    print(f"Memory usage before GC:")
    get_gpu_memory()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory usage after GC:")
    get_gpu_memory()


def get_run_name():
    return wandb.run.name if wandb.run else "offline-run"