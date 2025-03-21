import torch
import os
import json
import ast
import pandas as pd
import numpy as np

from code.utils.utils import clean_distractor, clean_question, clean_str, clean_str_punct_end


def load_df(filename, folder, nrows=None, other_obj_cols=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    obj_cols = other_obj_cols or []
    obj_cols = [col for col in obj_cols if col in df.columns]
    for i, row in df.iterrows():
        for col in obj_cols:
            try:
                df.at[i, col] = ast.literal_eval(row[col])
            except Exception:
                df.at[i, col] = None
    return df


def clean_text_df(df):
    df = df.fillna("")
    # Replace multiple newlines with a single newline, replace other whitespace with space
    df["question"] = df["question"].apply(lambda x: clean_question(x))
    # Do string cleaning before training, not during batch collation. Remove newlines in question.
    col_nms = ["option", "lvl_1_subject_name", "lvl_2_subject_name", "lvl_3_subject_name", "construct_name", "correct_answer", "misconception_name"]
    for col_nm in col_nms:
        df[col_nm] = df[col_nm].apply(lambda x: clean_str(x))
    col_nms = ["explanation", "solution"]
    for col_nm in col_nms:
        df[col_nm] = df[col_nm].apply(lambda x: clean_str_punct_end(x))
    # Replace back to nan since we might drop rows with nan in misconception_name
    df.replace("", np.nan, inplace=True)

    return df


def unlabel_samples(df: pd.DataFrame, percent_drop_labels: int):
    labeled_idxs = df["misconception_name"].notna()
    num_labeled_idxs = labeled_idxs.sum()
    p_array = np.full((len(df),), 1 / num_labeled_idxs)
    p_array[~labeled_idxs] = 0
    k = int(num_labeled_idxs * percent_drop_labels / 100)
    drop_idxs = np.random.choice(len(df), k, replace=False, p=p_array)
    df["misconception_name"].iloc[drop_idxs] = np.nan
    df["misconception_id"].iloc[drop_idxs] = np.nan


def load_data_finetune(configs, mode="with_error_label"):
    data = {}
    for name in ["train", "val", "test"]:
        df = load_df(f"{name}_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv", configs.data_dir)
        df = clean_text_df(df)
        # Drop rows without distractors
        df = df.loc[df["option"].notna()]
        if( name in ("train", "val") and configs.percent_drop_labels ):
           unlabel_samples(df, configs.percent_drop_labels)
        if( mode == "with_error_label" or configs.exclude_unlabeled_samples ):
            # Drop rows without error labels (misconception_name) for finetuning LM q(e|s,d) or p(e|s) or p(d|s,e) or baseline p(e,d|s)
            df = df.loc[df["misconception_name"].notna()]
        if( name == "test" and ( configs.task_name in ["d_given_s", "d_given_s_e", "e_given_s", "e_d_given_s", "e_given_s_d"] ) ):
            df = process_test_set(df, configs)
        json_out = df.to_json(orient="records")
        data[name] = json.loads(json_out)

    # Debug with less data
    if(configs.debug):
        data["train"] = data["train"][:4]
        data["val"] = data["val"][:4]
        data["test"] = data["test"][:3]

    print(f"Loaded {len(data['train'])} train samples, {len(data['val'])} val samples, {len(data['test'])} test samples/questions.")

    return data["train"], data["val"], data["test"]


def load_data_distractorgen(configs):
    data = {}
    for name in ["train", "val", "test"]:
        df = load_df(f"{name}_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv", configs.data_dir)
        df = clean_text_df(df)
        # Drop rows without distractors
        df = df.loc[df["option"].notna()]
        if configs.exclude_unlabeled_samples:
            df = df.loc[df["misconception_name"].notna()]
        if name in ("train", "val") and configs.percent_drop_labels:
            unlabel_samples(df, configs.percent_drop_labels)
        data[name] = df

    data["train_without_error_label"] = data["train"].loc[data["train"]["misconception_name"].isna()]
    data["train_with_error_label"] = data["train"].loc[data["train"]["misconception_name"].notna()]
    del data["train"]

    data["test_with_error_label"] = data["test"].loc[data["test"]["misconception_name"].notna()]
    # Transform test set: add ground truth distractors for each input question
    data["test_with_error_label"] = process_test_set(data["test_with_error_label"], configs, drop_if_less_err=True)
    data["test"] = process_test_set(data["test"], configs)

    for name in ["train_without_error_label", "train_with_error_label", "val", "test", "test_with_error_label"]:
        json_out = data[name].to_json(orient="records")
        data[name] = json.loads(json_out)

    # Debug with less data
    if(configs.debug):
        data["train_without_error_label"] = data["train_without_error_label"][:4]
        data["train_with_error_label"] = data["train_with_error_label"][:4]
        data["val"] = data["val"][:3]
        data["test"] = data["test"][:3]
        data["test_with_error_label"] = data["test_with_error_label"][:3]

    print(f"Loaded {len(data['train_without_error_label'])} train samples without error labels, {len(data['train_with_error_label'])} train samples with error labels, {len(data['val'])} val samples, {len(data['test'])} test questions, {len(data['test_with_error_label'])} test samples/questions with error labels.")

    return data["train_without_error_label"], data["train_with_error_label"], data["val"], data["test"], data["test_with_error_label"]


def process_test_set(df, configs, drop_if_less_err=False, drop_if_no_err=False, clean_distractors=True):
    # Clean distractor only in test set for evaluation. Raw text is used for training. 
    # Don't clean correct answer here since we need the raw text in the test prompts to match train prompt style. We clean correct answer on the fly when we need to remove generated distractors equal to correct answer post cleaning.
    if clean_distractors:
        df["option"] = df["option"].apply(lambda x: clean_distractor(x))

    # Drop duplicate ground truth distractors for each question. These duplicate ground truth distractors map to different misconceptions, we keep the first misconception label.
    df.drop_duplicates(subset=["qid", "option_idx"], keep='first', inplace=True)

    # Pool ground truth distractors for each question
    g = df.groupby("qid")
    list_cols = ["option", "option_idx", "explanation", "misconception_id", "misconception_name", "proportion"]
    df = g.agg({col: lambda x: list(x) for col in list_cols}).join(g[["question", "lvl_1_subject_id", "lvl_1_subject_name", "lvl_2_subject_id", "lvl_2_subject_name", "lvl_3_subject_id", "lvl_3_subject_name", "construct_id", "construct_name", "correct_answer", "solution"]].nth(0)).reset_index()

    if(
        ( configs.exp_name == "pretrain" and ( configs.task_name in ["d_given_s_e", "e_given_s", "e_d_given_s"] ) )
        or configs.exclude_unlabeled_samples or drop_if_less_err
    ):
        # Drop questions with less than 3 errors in test set
        df["misconception_name_count"] = df["misconception_name"].apply(lambda x: len(x))
        df = df.loc[df["misconception_name_count"] >= 3]
        # Keep first three ground truth errors for each question
        df["misconception_name"] = df["misconception_name"].apply(lambda x: x[:3])
    elif( drop_if_no_err ):
        df["misconception_name_count"] = df["misconception_name"].apply(lambda x: len([err for err in x if pd.notna(err)]))
        df = df.loc[df["misconception_name_count"] >= 1]
        # Keep first three ground truth errors for each question
        df["misconception_name"] = df["misconception_name"].apply(lambda x: x[:3])

    message = "Number of questions in test set with error label:" if drop_if_less_err else "Number of questions in test set:"
    print(message, len(df))
    # Assert 3 distractors per question
    assert df["option"].apply(len).eq(configs.num_distractors_per_question).all(), "Number of distractors per question is not equal to 3."

    return df


def get_data_loaders_finetune(train_set, val_set, collate_fn, tokenizer, device, configs):
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn(tokenizer, device, configs), 
                                                batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=True, drop_last=False)                        
    val_loader = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn(tokenizer, device, configs), 
                                                batch_size=configs.test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


def get_test_data_loader_finetune(test_set, collate_fn, tokenizer, device, configs):
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn(tokenizer, device, configs), 
                                                batch_size=configs.test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)

    return test_loader


def get_data_loaders_distractorgen(train_set_without_error_label, train_set_with_error_label, val_set, collator_without_err_label, collator_with_err_label, configs):
    train_loader_without_error_label = torch.utils.data.DataLoader(train_set_without_error_label, collate_fn=collator_without_err_label,
                                                batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=bool(train_set_without_error_label), drop_last=False)
    train_loader_with_error_label = torch.utils.data.DataLoader(train_set_with_error_label, collate_fn=collator_with_err_label,
                                                batch_size=configs.batch_size * configs.num_monte_carlo_samples, num_workers=configs.num_workers,
                                                shuffle=bool(train_set_with_error_label), drop_last=False) # Turn off shuffle if no samples to avoid crashing
    val_loader = torch.utils.data.DataLoader(val_set, collate_fn=collator_without_err_label,
                                                batch_size=configs.test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)

    return train_loader_without_error_label, train_loader_with_error_label, val_loader


def get_test_data_loader_distractorgen(test_set, collate_fn, tokenizer, device, configs, pipeline, test_batch_size):
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn(tokenizer, device, configs, pipeline), 
                                                batch_size=test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)

    return test_loader
