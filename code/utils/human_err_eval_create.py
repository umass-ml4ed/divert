import random
import hydra
import re
import pandas as pd

from code.utils.load_data import load_df, load_data_distractorgen
from code.utils.utils import save_csv, set_random_seed

img_re = re.compile(r".*!\[.*\].*\(.*\)")

def sample_qids(df: pd.DataFrame, num_qids: int):
    df["has_img"] = df.apply(lambda row: bool(img_re.match(row["question"]) or img_re.match(row["correct_answer"])), axis=1)
    df = df[~df["has_img"]]
    df = df.sample(len(df), replace=False)
    # Bad latex, which of the following, unclear, Tom and Katie, which of the following, bad latex
    bad_qids = [103709, 106593, 104953, 131515, 89405, 78239]
    # Removed because too repetitive with previously selected questions
    bad_qids += [120903, 146281, 133673]
    df = df[~df["qid"].isin(bad_qids)]
    # Picked for new topics to replace previous repetitive ones
    hard_qids = [102233, 101551, 107347]
    df = pd.concat([df.iloc[:num_qids - len(hard_qids)], df[df["qid"].isin(hard_qids)]])
    return df

def format_question(row: pd.Series):
    return f"Question: {row['question']}\n" +\
        f"Correct Answer: {row['correct_answer']}\n" +\
        f"Solution: {row['solution']}"

def add_stop_token(text: str):
    return text.strip(".") + "."

@hydra.main(version_base=None, config_path="../distractorgen/", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)

    _, _, _, _, test_df = load_data_distractorgen(configs) # Just get the labeled subset of the test set
    test_df = pd.DataFrame(test_df)
    test_df = sample_qids(test_df, 20)
    test_df["question_formatted"] = test_df.apply(format_question, axis=1)
    test_df["rating"] = ""

    methods = [
        ("vae", "predicted_error_p_e_given_s", "vae_sample-e-False_num-e-10_divbeam-e-True_fdbk-False_rsn-False_seed-21_metamath-mistral-7b_jolly-capybara-1331_e_given_s.csv"),
        ("baseline", "predicted_error", "baseline_e-given-s_sample-e-False_num-e-10_divbeam-e-True_fdbk-False_rsn-False_seed-21_metamath-mistral-7b_misty-elevator-1136.csv"),
        ("gpt-4o", "predicted_error_p_e_given_s", "gpt-4o_errors.csv")
    ]
    method_to_qid_to_errs = {}
    for name, err_col, filename in methods:
        method_df = load_df(filename, configs.results_dir)
        method_df = method_df.drop_duplicates(["qid", err_col])
        method_df = method_df.groupby("qid").head(3).reset_index(drop=True)
        method_to_qid_to_errs[name] = {}
        for _, row in method_df.iterrows():
            errs = method_to_qid_to_errs[name].setdefault(row["qid"], [])
            errs.append(row[err_col])
    method_to_qid_to_errs["gt"] = {row["qid"]: [
        add_stop_token(err) for err in row["misconception_name"]] for _, row in test_df.iterrows()}
    def get_errors(row: pd.Series):
        errors = list({err for qid_to_errs in method_to_qid_to_errs.values() for err in qid_to_errs[row["qid"]]})
        random.shuffle(errors)
        return errors
    test_df["error"] = test_df.apply(get_errors, axis=1)
    test_df = test_df.explode("error")
    test_df["methods"] = test_df.apply(lambda row: [
        method for method, qid_to_errs in method_to_qid_to_errs.items()
        if row["error"] in qid_to_errs[row["qid"]]
    ], axis=1)

    anno_df = test_df[["question_formatted", "error", "rating"]]
    anno_df = anno_df.rename(columns={"question_formatted": "question"})
    save_csv(test_df, "human_eval_ref_v2", configs.results_dir)
    save_csv(anno_df, "error_evaluation_v2", configs.results_dir)

if __name__ == '__main__':
    main()
