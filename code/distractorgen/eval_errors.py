import hydra
import itertools
import numpy as np
import pandas as pd
import evaluate
from sentence_transformers import SentenceTransformer, util
import logging
import os
import re

from code.utils.load_data import load_df
from code.utils.data_utils import get_question_plus_meta_data
from code.utils.utils import set_random_seed, clean_str
from code.utils.openai_api import OpenAIClient


SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
rouge_metric = evaluate.load("rouge", seed=21)
bertscore_metric = evaluate.load("bertscore", seed=21)


def sentence_similarity_sbert(sentence1, sentence2):
    if(type(sentence1) == float):
        sentence1 = ""
    if(type(sentence2) == float):
        sentence2 = ""
    emb1 = model.encode(sentence1, convert_to_tensor=True)
    emb2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2)

    return cosine_scores.item()


def sentence_similarity_rouge(sentence1, sentence2):
    if(type(sentence1) == float):
        sentence1 = ""
    if(type(sentence2) == float):
        sentence2 = ""
    rougescore = np.array(rouge_metric.compute(predictions=[sentence1], references=[sentence2], use_stemmer=True, use_aggregator=False)["rougeL"])

    return rougescore.item()


def sentence_similarity_bertscore(sentence1, sentence2):
    if(type(sentence1) == float):
        sentence1 = ""
    if(type(sentence2) == float):
        sentence2 = ""
    bertscore = np.array(bertscore_metric.compute(predictions=[sentence1], references=[sentence2], model_type="microsoft/deberta-xlarge-mnli")["f1"])

    return bertscore.item()


def compute_diversity(errs):
    sentence_similarity = sentence_similarity_sbert
    sim_score_errs = []
    for err in errs:
        assert type(err) == list, f"Errors for question is not a list: {err}"
        err_pairs = list(itertools.combinations(err, 2))
        sim_score_err_pairs = []
        for err_pair in err_pairs:
            sim_score = sentence_similarity(err_pair[0], err_pair[1])
            sim_score_err_pairs.append(sim_score)
        if( len(sim_score_err_pairs) == 0 ):
            pass
        else:
            sim_score_err = np.mean(np.asarray(sim_score_err_pairs, dtype=np.float32))
            sim_score_errs.append(sim_score_err)
    sim_score_errs_mean = np.mean(np.asarray(sim_score_errs, dtype=np.float32))
    div_score = 1.0 - sim_score_errs_mean

    return div_score


def compute_similarity_match_one(pred_errs, gt_errs, permute=True):
    sentence_similarity = sentence_similarity_sbert
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    sim_score_errs_max = []
    for pred_err, gt_err in zip(pred_errs, gt_errs):
        if( permute ):
            gt_err_perms = list(itertools.permutations(gt_err))
        else:
            # For evaluation e|s,d we don't permute ground truth errors since both pred err and gt err are ordered wrt distractors in the input
            gt_err_perms = [gt_err]
        sim_score_err_perms = []
        for gt_err_perm in gt_err_perms:
            sim_score_err_perm = []
            for pred_err_i, gt_err_i in zip(pred_err, gt_err_perm):
                sim_score = sentence_similarity(pred_err_i, gt_err_i)
                sim_score_err_perm.append(sim_score)
            sim_score_err_perm_mean = np.mean(np.asarray(sim_score_err_perm, dtype=np.float32))
            sim_score_err_perms.append(sim_score_err_perm_mean)
        sim_score_err_max = np.max(np.asarray(sim_score_err_perms, dtype=np.float32))
        sim_score_errs_max.append(sim_score_err_max)
    sim_score_errs_mean_max = np.mean(np.asarray(sim_score_errs_max, dtype=np.float32))

    return sim_score_errs_mean_max


def gpt4_error_comparison_prompt(question, err_1, err_2):
    return "You are a math education expert. " +\
        "Your job is to determine if two error descriptions are mathematically equivalent for a given question. " +\
        "After seeing the two errors, provide a brief one sentence reasoning for why they may or may not be equivalent. " +\
        "Then indicate your final answer with \"Answer: Equivalent\" or \"Answer: Not Equivalent\".\n\n" +\
        get_question_plus_meta_data(question) + "\n\n" +\
        f"The first error is: {err_1}\n\n" +\
        f"The second error is: {err_2}"


def gpt4_extract_result(result: str):
    return 1 if re.findall(r"Answer: Equivalent", result) else 0


def gpt4_error_similarities(df: pd.DataFrame, error_pairs):
    prompts = [
        gpt4_error_comparison_prompt(df.iloc[qidx], err_1, err_2)
        for qidx, err_1, err_2 in error_pairs
    ]
    client = OpenAIClient(True)
    responses = client.get_batched_responses(prompts, "4o", 400, 20, 0, show_progress=True)
    return {key: gpt4_extract_result(resp) for key, resp in zip(error_pairs, responses)}


def compute_similarity_match_any(df, errs_list_1, errs_list_2, configs):
    # Get error pairs and compute their similarity scores
    error_pairs = [
        (qidx, err_1, err_2)
        for qidx, (err_list_1, err_list_2) in enumerate(zip(errs_list_1, errs_list_2))
        for err_1 in err_list_1 for err_2 in err_list_2
    ]
    if configs.error_similarity == "gpt4":
        similarities = gpt4_error_similarities(df, error_pairs)
    else:
        if configs.error_similarity == "sbert":
            sim_fn = sentence_similarity_sbert
        elif configs.error_similarity == "rouge":
            sim_fn = sentence_similarity_rouge
        elif configs.error_similarity == "bertscore":
            sim_fn = sentence_similarity_bertscore
        else:
            raise f"Invalid scoring function {configs.error_similarity}"
        similarities = {
            (qidx, err_1, err_2): sim_fn(err_1, err_2)
            for qidx, err_1, err_2 in error_pairs
        }

    # Get error-level similarity based on pair similarities
    sim_score_errs_max = []
    for qidx, (err_list_1, err_list_2) in enumerate(zip(errs_list_1, errs_list_2)):
        for err_1 in err_list_1:
            sim_score_errs_max.append(np.max(
                np.asarray([similarities[(qidx, err_1, err_2)] for err_2 in err_list_2], dtype=np.float32)
            ))

    # Overall mean is equal to question level mean since we ensure each question has same number (3) of errors
    sim_score_errs_mean_max = np.mean(np.asarray(sim_score_errs_max, dtype=np.float32))
    return sim_score_errs_mean_max


def eval_errors(df, pred_err_col_nm, configs, filename, mode="match_any", permute=True):
    print(f"--- Evaluating errors ---")

    # Keep top-3 errors for each question, errors are already sorted by score in descending order
    df[pred_err_col_nm] = df[pred_err_col_nm].apply(lambda x: clean_str(str(x)))
    df = df.groupby("qid").head(3)
    g = df.groupby("qid")
    df = g.agg({pred_err_col_nm: lambda x: list(x)}).join(g[[col for col in df.columns if col not in ["qid", pred_err_col_nm]]].nth(0)).reset_index()

    print(f"Num questions for error eval: {len(df)}")
    if(len(df) == 0):
        print("No questions for error eval")
        pred_errs_div = None
        gt_errs_div = None
        sim_score_match_one = None
        sim_score_match_any_precision = None
        sim_score_match_any_recall = None
        sim_score_match_any_f1 = None
    else:
        pred_errs = df[pred_err_col_nm].tolist()
        gt_errs = df["misconception_name"].tolist()
        
        logging.disable(logging.INFO)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        # Compute diversity among predicted and ground truth errors
        pred_errs_div = compute_diversity(pred_errs)
        gt_errs_div = compute_diversity(gt_errs)
        print(f"Predicted errors diversity: {pred_errs_div}")
        print(f"Ground truth errors diversity: {gt_errs_div}")

        # Compute similarity of predicted and ground truth errors
        if( mode == "match_one" ):
            sim_score_match_one = compute_similarity_match_one(pred_errs, gt_errs, permute)
            print(f"Predicted errors similarity (match one): {sim_score_match_one}")
        else:
            sim_score_match_any_precision = compute_similarity_match_any(df, pred_errs, gt_errs, configs)
            sim_score_match_any_recall = compute_similarity_match_any(df, gt_errs, pred_errs, configs)
            sim_score_match_any_f1 = 2 * (sim_score_match_any_precision * sim_score_match_any_recall) / (sim_score_match_any_precision + sim_score_match_any_recall)
            print(f"Predicted errors similarity precision (match any): {sim_score_match_any_precision}")
            print(f"Predicted errors similarity recall (match any): {sim_score_match_any_recall}")
            print(f"Predicted errors similarity f1 (match any): {sim_score_match_any_f1}")
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    print(f"--- Done evaluating errors ---")

    results_dir = configs.results_dir + "/debug/" if configs.debug else configs.results_dir
    with open(f"{results_dir}/{filename}.txt", "w") as f:
        f.write(f"Num questions for error eval: {len(df)}\n")
        f.write(f"Num (s,d) samples for error eval: {len(df) * 3}\n")
        f.write(f"Predicted errors diversity: {pred_errs_div}\n")
        f.write(f"Ground truth errors diversity: {gt_errs_div}\n")
        if( mode == "match_one" ):
            f.write(f"Predicted errors similarity (match one): {sim_score_match_one}\n")
        else:
            f.write(f"Predicted errors similarity precision (match any): {sim_score_match_any_precision}\n")
            f.write(f"Predicted errors similarity recall (match any): {sim_score_match_any_recall}\n")
            f.write(f"Predicted errors similarity f1 (match any): {sim_score_match_any_f1}\n")