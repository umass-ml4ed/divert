from transformers import AutoTokenizer
import hydra
import pandas as pd

from code.utils.load_data import load_data_finetune


def get_token_count(df, configs):
    df = df.fillna("")
    col_names_pick_first = ["question"]
    # Comment next two lines for column name "option" or "misconception_name" since we don't group by question id
    g = df.groupby("qid")
    df = g[col_names_pick_first].nth(0).reset_index()
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
    for col_name in col_names_pick_first:
        tokenized = tokenizer(df[col_name].tolist(), padding=False, truncation=False, add_special_tokens=False)
        num_tokens = [len(token_list) for token_list in tokenized["input_ids"]]
        df[f"num_tokens_{col_name}"] = num_tokens
        print(df[f"num_tokens_{col_name}"].mask(df[f"num_tokens_{col_name}"] == 0).describe())


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    mode = "with_error_label"
    train_set, val_set, test_set = load_data_finetune(configs, mode)
    df_train = pd.DataFrame(train_set)
    df_val = pd.DataFrame(val_set)
    df_test = pd.DataFrame(test_set)
    # Use this line for column name "option" or "misconception_name" since test set is grouped by question id
    #df_test = df_test.explode(["option", "misconception_id", "misconception_name"])
    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        print(f"\n->{name} data\n")
        get_token_count(df, configs)


if __name__ == '__main__':
    main()