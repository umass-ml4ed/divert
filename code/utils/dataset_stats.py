import hydra
import pandas as pd

from code.utils.load_data import load_data_distractorgen

@hydra.main(version_base=None, config_path="../distractorgen/", config_name="configs")
def main(configs):
    configs.exclude_unlabeled_samples = False
    train_without_err, train_with_err, val, test, _ = load_data_distractorgen(configs)
    train = pd.concat([pd.DataFrame(train_without_err), pd.DataFrame(train_with_err)])
    val = pd.DataFrame(val)
    test = pd.DataFrame(test)
    test = test.explode(["option", "option_idx", "explanation", "misconception_id", "misconception_name", "proportion"])
    full_set = pd.concat([train, val, test])

    total_len = len(full_set)
    train_len = len(train)
    val_len = len(val)
    test_len = len(test)
    print("Sample-level")
    print(f"Train: {100 * train_len / total_len:.2f} ({train_len} / {total_len})")
    print(f"Val: {100 * val_len / total_len:.2f} ({val_len} / {total_len})")
    print(f"Test: {100 * test_len / total_len:.2f} ({test_len} / {total_len})")

    total_qids = len(full_set["qid"].unique())
    train_qids = len(train["qid"].unique())
    val_qids = len(val["qid"].unique())
    test_qids = len(test["qid"].unique())
    print("Question-level")
    print(f"Train: {100 * train_qids / total_qids:.2f} ({train_qids} / {total_qids})")
    print(f"Val: {100 * val_qids / total_qids:.2f} ({val_qids} / {total_qids})")
    print(f"Test: {100 * test_qids / total_qids:.2f} ({test_qids} / {total_qids})")

    print(f"Errors: {len(full_set['misconception_name'].unique())}")

    print("Topics:")
    print(f"Level 1: {len(full_set['lvl_1_subject_id'].unique())}")
    print(f"Level 2: {len(full_set['lvl_2_subject_id'].unique())}")
    print(f"Level 3: {len(full_set['lvl_3_subject_id'].unique())}")
    print(f"Construct: {len(full_set['construct_id'].unique())}")

    print("Errors:")
    print(f"Num unique errors: {len(full_set['misconception_name'].unique())}")

    print(f"Num s-d pairs with error labels: {100 * len(full_set.loc[full_set['misconception_name'].notna()]) / len(full_set):.2f}")


if __name__ == "__main__":
    main()
