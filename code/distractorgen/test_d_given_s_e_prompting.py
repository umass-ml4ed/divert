"""
Test distractor generation consistency via prompting
"""

import hydra
import re
from transformers import logging

from code.utils.openai_api import OpenAIClient
from code.utils.utils import set_random_seed, sanitize_configs, clean_distractor
from code.utils.load_data import load_data_distractorgen
from code.utils.data_utils import get_question_plus_meta_data


def create_prompt(question: dict, dis_idx: int):
    return "You are a math education expert. " +\
        "Your job is to write the incorrect answer that corresponds to a textual error description for a math question. " +\
        "Indicate your answer with the template \"The incorrect answer is: <answer>\".\n\n" +\
        get_question_plus_meta_data(question) + "\n\n" +\
        f"The error is: {question['misconception_name'][dis_idx]}"


def extract_distractor(resp: str):
    match = re.match(r"The incorrect answer is: (.*)", resp)
    if match:
        return match.group(1)
    return ""


def test(test_set_with_err):
    """
    Calculate the portion of *unique* qid/error pairs where a corresponding distractor is correctly generated
    """
    prompts = [
        create_prompt(question, dix_idx) for question in test_set_with_err for dix_idx in range(3)
    ]
    client = OpenAIClient(True)
    # Generate predicted distractors for each qid/err in the data using greedy decoding
    responses = client.get_batched_responses(prompts, "4o", 400, 20, 0, show_progress=True)
    pred_distractors = [extract_distractor(resp) for resp in responses]
    gt_distractors = [question['option'][dis_idx] for question in test_set_with_err for dis_idx in range(3)]
    matched = [clean_distractor(gt_dis) == clean_distractor(pred_dis) for gt_dis, pred_dis in zip(gt_distractors, pred_distractors)]
    # Normalize by number of unique qid/err pairs
    # This ensures that matching *any* distractor (up to out of 3) from the set of a unique qid/err pair records a single match for that pair
    num_errors = sum([len(set(question["misconception_name"])) for question in test_set_with_err])
    print(f"Matched: {sum(matched) / num_errors}")


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Turn off warnings
    logging.set_verbosity_error()
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    configs.log_wandb = False # Will crash if set to true since no associated run
    # Load data
    _, _, _, _, test_set_with_err = load_data_distractorgen(configs)

    test(test_set_with_err)


if __name__ == "__main__":
    main()
