import re
import pandas as pd
import hydra

from code.distractorgen.eval_errors import eval_errors
from code.utils.utils import set_random_seed, save_csv
from code.utils.load_data import load_df

error_line_re = re.compile(r"(?i)distractor ?(?:\d+) error: (.+)")

def extract_errors(raw_response: str, num_distractor_samples: int):
    errors = []
    for line in raw_response.split("\n"):
        match = error_line_re.match(line)
        if match:
            errors.append(match.group(1).strip())
        if len(errors) == num_distractor_samples:
            break
    while len(errors) < num_distractor_samples:
        errors.append("(empty)")
    return errors

def reformat(df: pd.DataFrame, num_distractor_samples: int):
    ref_df = pd.read_csv("data/eedi_test_20_cleaned_4_18.csv")
    df["qid"] = ref_df["id"]
    df["predicted_error_p_e_given_s"] = df["raw_response"].apply(lambda x: extract_errors(x, num_distractor_samples))
    df = df.explode("predicted_error_p_e_given_s")
    df = df[["qid", "predicted_error_p_e_given_s"]]
    return df

@hydra.main(version_base=None, config_path="../distractorgen", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)

    # Load error-distractor generations
    df = load_df(f"zero_shot_all_info_error_ndis{configs.num_distractor_samples}_4o_none_q.csv", "results")
    df = reformat(df, configs.num_distractor_samples)
    save_csv(df, f"gpt-4o_errors_{configs.num_distractor_samples}", configs.results_dir)

    # configs.exclude_unlabeled_samples = True
    # eval_errors(df, "predicted_error_p_e_given_s", configs, "gpt-4o-zs-errors")

if __name__ == "__main__":
    main()
