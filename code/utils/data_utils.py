from code.utils.utils import clean_str, remove_full_stop


# Use double newlines as delimiter to separate different parts of the prompt
DELIM = "\n\n"


def get_error(item, configs, mode="without_error_label"):
    error = None
    if( configs.exp_name == "pretrain" ):
        error = item["misconception_name"]
    elif( configs.exp_name == "distractorgen" ):
        if( configs.testing ):
            if( mode == "with_error_label" ):
                # Use ground truth error to test consistency of p(d|s,e) model
                error = item["misconception_name"]
            else:
                error = item["predicted_error_p_e_given_s"]
        elif( configs.validation ):
            error = item['predicted_error']
        else:
            if( mode == "without_error_label" ):
                error = item['predicted_error']
            elif( mode == "with_error_label" ):
                error = item["misconception_name"]
    # Remove full stop if any at end of error since full stop is used as EOS token
    error = remove_full_stop(error)
    assert error != None, "Error is None"

    return error


def get_question_topic(item):
    # Yaya
    topic = "Math"
    if( item["lvl_1_subject_name"] is not None ):
        topic = item["lvl_1_subject_name"]
    if( item["lvl_2_subject_name"] is not None ):
        topic = item["lvl_2_subject_name"]
    if( item["lvl_3_subject_name"] is not None ):
        topic = item["lvl_3_subject_name"]

    return topic


def get_question_plus_meta_data(item):
    txt = f"The question is: {item['question']}{DELIM}" +\
            f"The question topic is: {get_question_topic(item)}{DELIM}" +\
            f"The question concept is: {item['construct_name']}{DELIM}" +\
            f"The solution is: {item['solution']}{DELIM}" +\
            f"The correct answer is: {item['correct_answer']}"

    return txt


def create_prompt_e_given_s_d(item, configs, mode="without_error_label", use_predicted_distractor=False):
    distractor = item["predicted_distractor" if use_predicted_distractor else "option"]
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
                f"{get_question_plus_meta_data(item)}{DELIM}" +\
                f"The incorrect answer given by the student is: {distractor}{DELIM}" +\
                f"The error made by the student is:"

    return prompt


def create_completion_e_given_s_d(item, _tokenizer, configs, mode="without_error_label", use_predicted_error=False):
    # Errors always end with full stop. Additional full stops are not present within errors, i.e., only a single full stop is present at the end.
    # Preceding single space is part of error by design to match calculating error mask token indices (tokens > question tokens and <= question+error tokens which includes a space token as the first token after question tokens). Full stop at end is considered part of the error generated. Therefore completion is suffixed with full stop. Full stop is used as stop token during generation.
    if use_predicted_error:
        error = item["predicted_error_p_e_given_s"].strip(".")
    else:
        error = get_error(item, configs, mode)
    completion = f" {error}."

    return completion


def create_prompt_e_given_s(item, _configs, mode="without_error_label"):
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"A possible error made by a student is:"

    return prompt


def create_completion_e_given_s(item, _tokenizer, configs, mode="without_error_label"):
    # Errors always end with full stop. Additional full stops are not present within errors, i.e., only a single full stop is present at the end.
    # Preceding single space is part of error. Full stop at end is considered part of the error generated. Therefore completion is suffixed with full stop. Full stop is used as stop token during generation.
    error = get_error(item, configs, mode)
    completion = f" {error}."

    return completion


def create_prompt_e_d_given_s(item, _configs, mode="without_error_label"):
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"The error made by the student is:"

    return prompt


def create_completion_e_d_given_s(item, tokenizer, configs, mode="without_error_label"):
    error = get_error(item, configs, mode)
    completion = f" {error}.{DELIM}" +\
                f"The incorrect answer given by the student is: {item['option']}{tokenizer.eos_token}"

    return completion


def create_prompt_d_given_s_e(item, configs, mode="without_error_label"):
    # Use predicted error by q(e|s,d) if ground truth error tokens label (misconception name) is not present
    error = get_error(item, configs, mode)
    suffix = "The incorrect answer given by the student is:"
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"The error made by the student is: {error}.{DELIM}" +\
            f"{suffix}"

    return prompt


def create_completion_d_given_s_e(item, tokenizer, configs, _mode="without_error_label"):
    # Suffix completion with tokenizer.eos_token
    # Need to suffix distractor completion with tokenizer.eos_token since generation needs to include full stop/period (ex: decimal numbers).
    option = item["option"]
    completion = f" {option}{tokenizer.eos_token}"
    return completion


def create_prompt_d_given_s(item, configs, _mode="without_error_label"):
    suffix = "The incorrect answer given by the student is:"
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"{suffix}"

    return prompt


def create_completion_d_given_s(item, tokenizer, configs, _mode="without_error_label"):
    # Suffix completion with tokenizer.eos_token
    # Need to suffix distractor completion with tokenizer.eos_token since generation needs to include full stop/period (ex: decimal numbers).
    completion = f" {item['option']}{tokenizer.eos_token}"
    return completion


def create_question_d_given_s_e(item, _mode="without_error_label"):
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"The error made by the student is:"

    return prompt


def create_question_error_d_given_s_e(item, configs, mode="without_error_label"):
    # Full stop at end of error is considered part of the error.
    error = get_error(item, configs, mode)
    prompt = f"A teacher assigns the following math question to a class of middle school students.{DELIM}" +\
            f"{get_question_plus_meta_data(item)}{DELIM}" +\
            f"The error made by the student is: {error}."

    return prompt


def get_targets(test_set):
    targets = [item["misconception_name"] for item in test_set]

    return targets


def get_stop_token_info(task_name, tokenizer):
    if( task_name == "d_given_s_e" or task_name == "d_given_s" ):
        stop_token = tokenizer.eos_token
        stop_token_id = tokenizer.eos_token_id
    else:
        stop_token = "."
        # Add dummy text since tokenization for llama/mistral is contextual. Example: "." -> [1, 869] / [1, 842] but "dummytext." -> [1, 20254, 726, 29889] / [1, 24084, 772, 28723] for llama/mistral respectively. Llama3 might not be contextual since Example: "." -> [1, 869] and "dummytext." -> [1, 20254, 726, 29889] have same token id for stop token ".".
        stop_token_id = tokenizer("dummytext.", add_special_tokens=True).input_ids[-1]

    return stop_token, stop_token_id


def post_process_predictions(predictions_batch, prompts_len, stop_token, configs):
    processed_predictions = []
    for prediction, prompt_len in zip(predictions_batch, prompts_len):
        # Remove prompt
        prediction = prediction[prompt_len:]
        prediction = clean_str(prediction)
        # Trim prediction post stop token
        prediction = prediction.split(stop_token, 1)[0]
        # Add stop token (".") back to generated errors. We need "." since during vae training we treat "." as part of the error sampled from q(e|s,d)
        if( stop_token == "." ):
            prediction = prediction + stop_token
        processed_predictions.append(prediction)

    return processed_predictions
