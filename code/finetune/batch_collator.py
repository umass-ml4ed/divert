import torch

from code.utils.data_utils import (create_prompt_e_given_s_d, create_completion_e_given_s_d, create_prompt_e_given_s, create_completion_e_given_s, 
    create_prompt_d_given_s_e, create_completion_d_given_s_e, create_prompt_d_given_s, create_completion_d_given_s, create_prompt_e_d_given_s, create_completion_e_d_given_s)


def create_prompt(item, configs):
    name_func_map = {
        "e_given_s" : create_prompt_e_given_s,
        "e_given_s_d" : create_prompt_e_given_s_d,
        "d_given_s_e": create_prompt_d_given_s_e,
        "d_given_s": create_prompt_d_given_s,
        "e_d_given_s": create_prompt_e_d_given_s
        }
    prompt = name_func_map[configs.task_name](item, configs)

    return prompt


def create_completion(item, tokenizer, configs):
    name_func_map = {
        "e_given_s" : create_completion_e_given_s,
        "e_given_s_d" : create_completion_e_given_s_d,
        "d_given_s_e": create_completion_d_given_s_e,
        "d_given_s": create_completion_d_given_s,
        "e_d_given_s": create_completion_e_d_given_s
        }
    completion = name_func_map[configs.task_name](item, tokenizer, configs)

    return completion
             

class CollateWrapperGenerative():
    def __init__(self, tokenizer, device, configs):
        self.tokenizer = tokenizer
        self.ignore_index = -100 # Default ignore index in CrossEntropyLoss
        self.device = device
        self.configs = configs


    def __call__(self, batch):
        # Construct text
        prompts = [create_prompt(item, self.configs) for item in batch]
        examples = [f"{create_prompt(item, self.configs)}{create_completion(item, self.tokenizer, self.configs)}" for item in batch]

        # Tokenize
        prompts_tokenized = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.configs.max_length, add_special_tokens=True)
        examples_tokenized = self.tokenizer(examples, padding=True, truncation=True, max_length=self.configs.max_length, add_special_tokens=True, return_tensors='pt').to(self.device)

        # Construct labels
        labels = examples_tokenized["input_ids"].detach().clone()
        # Ignore pad tokens when computing loss
        labels = labels.masked_fill((examples_tokenized["attention_mask"] == 0), self.ignore_index)
        # Ignore prompt tokens when computing loss
        prompts_len = torch.tensor([len(prompt_tokenized_input_ids) for prompt_tokenized_input_ids in prompts_tokenized["input_ids"]]).to(self.device)
        range_tensor = torch.arange(examples_tokenized["input_ids"].size(1), device=self.device).unsqueeze(0)
        range_tensor = range_tensor.repeat(prompts_len.size(0), 1)
        mask_tensor = (range_tensor < prompts_len.unsqueeze(-1)) 
        labels[mask_tensor] = self.ignore_index

        if( self.configs.debug ):
            print(f"prompts: {prompts}")
            print(f"examples: {examples}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            print(f"examples_tokenized: {examples_tokenized}")
            print(f"labels: {labels}")
            for ids in examples_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))

        return {
            "input_ids": examples_tokenized["input_ids"].to(self.device),
            "attention_mask": examples_tokenized["attention_mask"].to(self.device),
            "labels": labels.to(self.device)
            }


class CollateWrapperGenerativeTest():
    def __init__(self, tokenizer, device, configs):
        self.tokenizer = tokenizer
        self.device = device
        self.configs = configs


    def __call__(self, batch):
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        # Construct text
        prompts = [create_prompt(item, self.configs) for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        # Tokenize
        prompts_tokenized = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.configs.max_length, add_special_tokens=True, return_tensors="pt")
        correct_answers = [item["correct_answer"] for item in batch]

        if( self.configs.debug ):
            print(f"prompts: {prompts}")
            print(f"prompts_len: {prompts_len}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            for ids in prompts_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))

        return {
            "inputs": prompts_tokenized.to(self.device),
            "prompts_len": prompts_len,
            "correct_answers": correct_answers
            }