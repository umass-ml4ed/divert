"""
Count number of tokens in each column of the dataset to get max sequence length for input/output of models.
"""

from transformers import AutoTokenizer
import hydra

from code.utils.load_data import load_df, clean_text_df


def get_token_count(df, configs):
    df = df.fillna("")
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
    col_names = ["question", "option", "lvl_1_subject_name", "lvl_2_subject_name", "lvl_3_subject_name", "construct_name", "correct_answer", "misconception_name", "explanation", "solution"]
    col_names_describe = ["option", "misconception_name", "explanation"]
    sum_max = 0
    for col_name in col_names:
        print(f"\n->Processing col_name: {col_name}\n")
        tokenized = tokenizer(df[col_name].tolist(), padding=False, truncation=False, add_special_tokens=False)
        num_tokens = [len(token_list) for token_list in tokenized["input_ids"]]
        df[f"num_tokens_{col_name}"] = num_tokens
        column_max = df[f"num_tokens_{col_name}"].mask(df[f"num_tokens_{col_name}"] == 0).max()
        sum_max += column_max
        if( col_name in col_names_describe ):
            print(df[f"num_tokens_{col_name}"].mask(df[f"num_tokens_{col_name}"] == 0).describe([.99]))
    print(f"\nSum of max tokens across columns: {sum_max}\n")


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    for name in ["train", "val", "test"]:
        df = load_df(f"{name}_data{'_topic' if configs.topic_generalization else ''}_{configs.cross_val_fold}.csv", configs.data_dir)
        df = clean_text_df(df)
        # Drop rows without distractors
        df = df.loc[df["option"].notna()]
        print(f"\n->{name} data\n")
        get_token_count(df, configs)


if __name__ == '__main__':
    main()