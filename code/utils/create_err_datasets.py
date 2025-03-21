import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

def expand_df(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        cur_qs = []
        for opt_idx, opt_letter in enumerate(["A", "B", "C", "D"]):
            if opt_idx == row["CorrectAnswer"] - 1:
                correct_answer = row[f"Answer {opt_letter} Text"]
                solution = row[f"Explanation{opt_idx + 1}"]
            else:
                cur_qs.append({
                    "qid": row["QuestionId"],
                    "option_idx": opt_idx + 1,
                    "question": row["Question Text"],
                    "option": row[f"Answer {opt_letter} Text"],
                    "explanation": row[f"Explanation{opt_idx + 1}"],
                    "lvl_1_subject_id": row["Level1SubjectId"],
                    "lvl_1_subject_name": row["Level1SubjectName"],
                    "lvl_2_subject_id": row["Level2SubjectId"],
                    "lvl_2_subject_name": row["Level2SubjectName"],
                    "lvl_3_subject_id": row["Level3SubjectId"],
                    "lvl_3_subject_name": row["Level3SubjectName"],
                    "construct_id": row["ConstructId"],
                    "construct_name": row["ConstructName"],
                    "proportion": row[f"Answer{opt_idx + 1}Proportion"]
                })
        for opt in cur_qs:
            opt["correct_answer"] = correct_answer
            opt["solution"] = solution
        results.extend(cur_qs)
    return pd.DataFrame(results)

def get_questions():
    # Get original questions
    question_metadata = pd.read_csv("data/question_metadata.csv")
    number_questions = pd.read_csv("data/number_questions.csv").drop(columns=["QuestionId"])
    question_metadata = question_metadata.merge(
        number_questions,
        left_on="QuestionId",
        right_on="RealQuestionId",
        how="inner"
    ).drop(columns=["RealQuestionId"])
    question_metadata["Level1SubjectId"] = 32
    question_metadata["Level1SubjectName"] = "Number"

    # Get new questions
    new_questions = pd.read_csv("data/2024-01-19/data/question.csv")
    new_questions = new_questions[new_questions["QuestionText"].notna()]
    new_questions = new_questions.rename(columns={
        "QuestionText": "Question Text",
        "ExplanationA": "Explanation1",
        "ExplanationB": "Explanation2",
        "ExplanationC": "Explanation3",
        "ExplanationD": "Explanation4",
        "AnswerAText": "Answer A Text",
        "AnswerBText": "Answer B Text",
        "AnswerCText": "Answer C Text",
        "AnswerDText": "Answer D Text"
    })
    new_questions["Answer1Proportion"] = None
    new_questions["Answer2Proportion"] = None
    new_questions["Answer3Proportion"] = None
    new_questions["Answer4Proportion"] = None

    # Apply subjects
    q_sub = pd.read_csv("data/2024-01-19/data/question-subject.csv").drop(columns=["SubjectType", "ParentSubjectId"])
    for lvl in range(1, 4):
        q_sub_lvl = q_sub[q_sub["SubjectLevel"] == lvl].drop(columns=["SubjectLevel"])
        q_sub_lvl = q_sub_lvl.rename(columns={"SubjectId": f"Level{lvl}SubjectId", "SubjectName": f"Level{lvl}SubjectName"})
        new_questions = new_questions.merge(q_sub_lvl, how="left", on="QuestionId")

    # Join original and new questions, prefer new format
    question_metadata = pd.concat([new_questions, question_metadata])
    question_metadata = question_metadata.drop_duplicates(subset=["QuestionId"])
    print(f"Total questions: {len(question_metadata)}")
    return question_metadata

def get_misconceptions():
    # Join distractor sets
    distractor_misconceptions = pd.read_csv("data/distractor_misconceptions.csv").rename(columns={"QuestionId": "DMQuestionId"})
    new_misconceptions = pd.read_csv("data/2024-01-19/data/misconception.csv").rename(columns={"QuestionId": "DMQuestionId"})
    distractor_misconceptions = pd.concat([distractor_misconceptions, new_misconceptions])
    distractor_misconceptions = distractor_misconceptions.drop_duplicates(subset=["DMQuestionId", "AnswerValue", "MisconceptionId"])
    print(f"Total misconceptions: {len(distractor_misconceptions)}")
    return distractor_misconceptions

def write_dataset_files(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                        distractor_misconceptions: pd.DataFrame, fold: int, suffix: str = ""):
    # Expand data so each distractor gets its own row, assign misconceptions, and export
    print("Fold:", fold + 1)
    for split, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        df = expand_df(df)
        df = df.merge(
            distractor_misconceptions,
            left_on=["qid", "option_idx"],
            right_on=["DMQuestionId", "AnswerValue"],
            how="left"
        )
        df = df.drop(columns=["DMQuestionId", "AnswerValue"])
        df = df.rename(columns={"MisconceptionId": "misconception_id", "MisconceptionName": "misconception_name"})
        filename = f"data/{split}_data{'_' + suffix if suffix else ''}_{fold + 1}.csv"
        df.to_csv(filename, index=False)
        print(f"Split: {split}, # of samples: {len(df)}, with misconceptions: {len(df[~df['misconception_id'].isna()])}")

def generate_splits(question_metadata_labeled: pd.DataFrame, question_metadata_unlabeled: pd.DataFrame,
                    distractor_misconceptions: pd.DataFrame, fold: int):
    # Shift data based on the fold
    labeled_pivot = int(len(question_metadata_labeled) * fold * .2)
    question_metadata_labeled = pd.concat([
        question_metadata_labeled[labeled_pivot:],
        question_metadata_labeled[:labeled_pivot]
    ])
    unlabeled_pivot = int(len(question_metadata_unlabeled) * fold * .2)
    question_metadata_unlabeled = pd.concat([
        question_metadata_unlabeled[unlabeled_pivot:],
        question_metadata_unlabeled[:unlabeled_pivot],
    ])

    # Do train/val/test split, sampling equally from questions with/without misconception labels
    train_portion = 0.65
    val_portion = 0.15
    train_data = pd.concat([
        question_metadata_labeled[:int(len(question_metadata_labeled) * train_portion)],
        question_metadata_unlabeled[:int(len(question_metadata_unlabeled) * train_portion)]
    ])
    val_data = pd.concat([
        question_metadata_labeled[
            int(len(question_metadata_labeled) * train_portion):
            int(len(question_metadata_labeled) * (train_portion + val_portion))],
        question_metadata_unlabeled[
            int(len(question_metadata_unlabeled) * train_portion):
            int(len(question_metadata_unlabeled) * (train_portion + val_portion))]
    ])
    test_data = pd.concat([
        question_metadata_labeled[int(len(question_metadata_labeled) * (train_portion + val_portion)):],
        question_metadata_unlabeled[int(len(question_metadata_unlabeled) * (train_portion + val_portion)):]
    ])

    # Write files for splits
    write_dataset_files(train_data, val_data, test_data, distractor_misconceptions, fold)

def split_by_question_and_topic(question_metadata_labeled: pd.DataFrame, distractor_misconceptions: pd.DataFrame):
    print("Splitting by question and topic")
    # Ensure that no topics overlap between train/val/test splits
    topic_col = "Level2SubjectName"
    dummy_data = np.zeros((len(question_metadata_labeled,)))
    sss = StratifiedGroupKFold(n_splits=5)
    for fold, (non_test_idx, test_idx) in enumerate(sss.split(dummy_data, dummy_data, question_metadata_labeled[topic_col])):
        test_data = question_metadata_labeled.iloc[test_idx]
        # Group split non-test data to get train and val data
        non_test_data = question_metadata_labeled.iloc[non_test_idx]
        inner_dummy_data = np.zeros((len(non_test_data,)))
        inner_sss = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = next(inner_sss.split(inner_dummy_data, inner_dummy_data, non_test_data[topic_col]))
        train_data = non_test_data.iloc[train_idx]
        val_data = non_test_data.iloc[val_idx]
        write_dataset_files(train_data, val_data, test_data, distractor_misconceptions, fold, "topic")

def split_by_question(question_metadata_labeled: pd.DataFrame, question_metadata_unlabeled: pd.DataFrame,
                      distractor_misconceptions: pd.DataFrame):
    print("Splitting by question")
    # Create splits for each fold
    for fold in range(5):
        generate_splits(question_metadata_labeled, question_metadata_unlabeled, distractor_misconceptions, fold)

def main(do_split_by_topic: bool):
    initialize_seeds(221)

    # Get raw data
    question_metadata = get_questions()
    distractor_misconceptions = get_misconceptions()

    # Find which questions have some misconceptions labeled
    dm_qids = distractor_misconceptions["DMQuestionId"].drop_duplicates()
    question_metadata = question_metadata.merge(
        dm_qids,
        left_on="QuestionId",
        right_on="DMQuestionId",
        how="left"
    ).sample(frac=1)
    question_metadata_labeled = question_metadata[~question_metadata["DMQuestionId"].isna()]
    question_metadata_unlabeled = question_metadata[question_metadata["DMQuestionId"].isna()]

    # Split data by question and by question/topic
    split_by_question_and_topic(question_metadata_labeled, distractor_misconceptions)
    split_by_question(question_metadata_labeled, question_metadata_unlabeled, distractor_misconceptions)

if __name__ == "__main__":
    main()
