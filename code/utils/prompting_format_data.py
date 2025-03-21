import hydra
import pandas as pd

from code.utils.load_data import process_test_set, clean_text_df

def with_default(val, default_val):
    return val if pd.notna(val) else default_val

def process_df(df: pd.DataFrame, split: str, configs):
    df = clean_text_df(df)
    df = df[df["correct_answer"].notna()]

    # For test keep only question with 3 errs, for train keep only questions with > 0 errs (unless drop_not_fully_labeled given)
    if split == "test" or configs.drop_not_fully_labeled:
        df = df[df["misconception_name"].notna()]
        kwargs = {"drop_if_less_err": True}
    else:
        kwargs = {"drop_if_no_err": True}

    # Aggregate distractors within questions
    df = process_test_set(df, configs, clean_distractors=False, **kwargs)
    print(f"{len(df)} questions")

    # Convert to expected format
    df["id"] = df["qid"]
    df["correct_option"] = df.apply(lambda row: {
        "option": row["correct_answer"], "explanation": row["solution"]
    }, axis=1)
    df["construct_info"] = df.apply(lambda row: {
        "construct1": [row["lvl_2_subject_id"], row["lvl_2_subject_name"]],
        "construct2": [with_default(row["lvl_3_subject_id"], 0), with_default(row["lvl_3_subject_name"], "")],
        "construct3": [row["construct_id"], row["construct_name"]]
    }, axis=1)
    df["distractors"] = df.apply(lambda row: [
        {"option_idx": option_idx, "option": option, "explanation": explanation,
         "proportion": with_default(proportion, 0), "misconception": with_default(misconception, "")}
        for option_idx, option, explanation, proportion, misconception in
            zip(row["option_idx"], row["option"], row["explanation"], row["proportion"], row["misconception_name"])
    ], axis=1)
    df = df[["id", "question", "correct_option", "construct_info", "distractors"]]

    return df

@hydra.main(version_base=None, config_path="../distractorgen", config_name="configs")
def main(configs):
    train_data = pd.read_csv(f"data/train_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv")
    val_data = pd.read_csv(f"data/val_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv")
    test_data = pd.read_csv(f"data/test_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv")
    pool_data = pd.concat([train_data, val_data])
    process_df(pool_data, "train", configs).to_csv("data/eedi_train_80_cleaned_4_18.csv", index=False)
    process_df(test_data, "test", configs).to_csv("data/eedi_test_20_cleaned_4_18.csv", index=False)

if __name__ == "__main__":
    main()
