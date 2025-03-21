from itertools import combinations
import pandas as pd
import numpy as np
import hydra
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, ttest_ind

from code.utils.load_data import load_df
from code.utils.utils import save_csv, set_random_seed

METHODS = ["vae", "baseline", "gpt-4o", "gt"]

@hydra.main(version_base=None, config_path="../distractorgen/", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)

    ref_df = load_df("human_eval_ref_v2.csv", configs.results_dir, other_obj_cols=["methods"])
    ref_df = ref_df[["question_formatted", "error", "methods"]].rename(columns={"question_formatted": "question"})

    method_to_rating_list = {method: [] for method in METHODS}
    rating_vecs = []

    anno_filenames = [] # NOTE: fill in filenames here
    for filename in anno_filenames:
        anno_df = load_df(filename, configs.results_dir)
        anno_df["rating"] = pd.to_numeric(anno_df["rating"], errors="coerce") # Some ratings are comments for entries deemed non-errors
        anno_df["rating"] = anno_df["rating"].fillna(1) # Override textual entries with lowest score
        assert anno_df["rating"].min() >= 1
        assert anno_df["rating"].max() <= 5
        anno_df = anno_df.merge(ref_df, on=["question", "error"])
        # anno_df = anno_df.sort_values(["question", "error"])
        rating_vecs.append(anno_df["rating"].to_numpy())
        save_csv(anno_df, filename.replace(".csv", "_ref"), configs.results_dir)

        for _, row in anno_df.iterrows():
            for method in row["methods"]:
                method_to_rating_list[method].append(row["rating"])

    for method, ratings in method_to_rating_list.items():
        ratings = np.array(ratings)
        print(f"{method}: {ratings.mean():.2f} \pm {ratings.std():.2f} ({ratings.sum()} / {len(ratings)})")

    kappa = cohen_kappa_score(rating_vecs[0], rating_vecs[1], weights="quadratic")
    correlation = pearsonr(rating_vecs[0], rating_vecs[1])
    print(f"QWK: {kappa:.4f}")
    print(f"Pearson Corr: {correlation}")
    for m1, m2 in combinations(METHODS, 2):
        stat = ttest_ind(method_to_rating_list[m1], method_to_rating_list[m2], equal_var=False)
        print(f"T-Test {m1} vs. {m2}: {stat}")

if __name__ == '__main__':
    main()
