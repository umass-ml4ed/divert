import torch
import random
import copy

from code.utils.data_utils import (create_prompt_e_given_s_d, create_completion_e_given_s_d, create_prompt_e_given_s, 
    create_completion_e_given_s, create_prompt_d_given_s_e, create_completion_d_given_s_e, create_question_d_given_s_e, 
    create_question_error_d_given_s_e, get_stop_token_info, post_process_predictions)


class CollateWrapperParent():
    def __init__(self, tokenizer, device, configs):
        self.tokenizer = tokenizer
        self.device = device
        self.configs = configs
        self.ignore_index = -100 # Default ignore index in CrossEntropyLoss


    def get_inputs(self, name, batch, create_prompt, create_completion, get_error_mask=True, mode="without_error_label", prompt_kwargs=None, completion_kwargs=None):
        prompt_kwargs = prompt_kwargs or {}
        completion_kwargs = completion_kwargs or {}
        prompts = [f"{create_prompt(item, self.configs, mode, **prompt_kwargs)}" for item in batch]
        examples = [f"{create_prompt(item, self.configs, mode, **prompt_kwargs)}{create_completion(item, self.tokenizer, self.configs, mode, **completion_kwargs)}" for item in batch]

        # Tokenize
        # Set add_special_tokens = true (default) since only BOS is prepended. EOS is not appended. This is fine since we don't want to pass errors with EOS token to the input of p(d|s,e) causing the input to have an EOS token in between.
        prompts_tokenized = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.configs.max_length, add_special_tokens=True)
        examples_tokenized = self.tokenizer(examples, padding=True, truncation=True, max_length=self.configs.max_length, return_tensors='pt', add_special_tokens=True).to(self.device)
        # Assumes no input is truncated by setting a large enough max_length
        assert examples_tokenized["input_ids"].shape[1] != self.configs.max_length, "Error: truncation might have occurred"

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

        if( get_error_mask ):
            if( name == "e_given_s" or name == "e_given_s_d" ):
                error_mask = (labels != self.ignore_index)
            else:
                questions = [f"{create_question_d_given_s_e(item, mode)}" for item in batch]
                questions_tokenized = self.tokenizer(questions, padding=False, truncation=True, max_length=self.configs.max_length, add_special_tokens=True)
                questions_len = torch.tensor([len(question_tokenized_input_ids) for question_tokenized_input_ids in questions_tokenized["input_ids"]]).to(self.device)
                questions_errors = [f"{create_question_error_d_given_s_e(item, self.configs, mode)}" for item in batch]
                questions_errors_tokenized = self.tokenizer(questions_errors, padding=False, truncation=True, max_length=self.configs.max_length, add_special_tokens=True)
                questions_errors_len = torch.tensor([len(question_error_tokenized_input_ids) for question_error_tokenized_input_ids in questions_errors_tokenized["input_ids"]]).to(self.device)

                error_labels = examples_tokenized["input_ids"].detach().clone()
                range_tensor = torch.arange(examples_tokenized["input_ids"].size(1), device=self.device).unsqueeze(0)
                range_tensor = range_tensor.repeat(questions_len.size(0), 1)
                mask_tensor = ((range_tensor < questions_len.unsqueeze(-1)) | (range_tensor >= questions_errors_len.unsqueeze(-1)))
                error_labels[mask_tensor] = self.ignore_index
                error_mask = (error_labels != self.ignore_index)

            inputs = {
                "input_ids": examples_tokenized["input_ids"].to(self.device),
                "attention_mask":examples_tokenized["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
                "error_mask": error_mask.to(self.device)
            }
        else:
            inputs = {
                "input_ids": examples_tokenized["input_ids"].to(self.device),
                "attention_mask":examples_tokenized["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }

        if( self.configs.debug ):
            print(f"name: {name}")
            print(f"prompts: {prompts}")
            print(f"examples: {examples}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            print(f"examples_tokenized: {examples_tokenized}")
            print(f"labels: {labels}")
            for ids in examples_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))

        return inputs


class CollateWrapperParentWithoutErrorLabel(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs, pipeline):
        super().__init__(tokenizer, device, configs)
        self.pipeline = pipeline
        self.temperature = 1.0


    def decay_softmax_temperature(self, decay_rate):
        self.temperature *= decay_rate


    def sample_errors_q_e_given_s_d(self, batch):
        # Sample e hat from q(e|s,d) with batched inference
        self.tokenizer.padding_side = "left" # Batched inference requires tokenizer padding side as left
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id

        prompts = [create_prompt_e_given_s_d(item, self.configs, mode="without_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
        stop_token, stop_token_id = get_stop_token_info("e_given_s_d", self.tokenizer)
        with self.pipeline.model_ctm("e_given_s_d_adapter", "eval"):
            # Nucleus sampling with top-k
            outputs = self.pipeline.model.generate(
                **encodings,
                max_new_tokens = self.configs.max_new_tokens_error,
                do_sample = self.configs.do_sample_errors_train,
                temperature = self.temperature,
                top_p = self.configs.top_p,
                top_k = self.configs.top_k,
                eos_token_id = stop_token_id
                )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = post_process_predictions(predictions, prompts_len, stop_token, self.configs)
        if( self.configs.debug ):
            # Add dummy text since GPT2 used for debugging generates blank e hat samples sometimes
            predictions = ["dummy error text" if len(prediction) == 0 else prediction for prediction in predictions]

        return predictions


    def sample_errors_q_ref_e_given_s_d(self, batch):
        # Sample e hat from q(e|s,d) with batched inference
        self.tokenizer.padding_side = "left" # Batched inference requires tokenizer padding side as left
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id

        prompts = [create_prompt_e_given_s_d(item, self.configs, mode="without_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
        stop_token, stop_token_id = get_stop_token_info("e_given_s_d_ref", self.tokenizer)
        with self.pipeline.model_ctm("e_given_s_d_ref_adapter", "eval"):
            # Nucleus sampling with top-k
            outputs = self.pipeline.model.generate(
                **encodings,
                max_new_tokens = self.configs.max_new_tokens_error,
                do_sample = self.configs.do_sample_errors_train,
                top_p = self.configs.top_p,
                top_k = self.configs.top_k,
                eos_token_id = stop_token_id
                )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = post_process_predictions(predictions, prompts_len, stop_token, self.configs)
        if( self.configs.debug ):
            # Add dummy text since GPT2 used during debugging generates blank e hat samples sometimes
            predictions = ["dummy error text" if len(prediction) == 0 else prediction for prediction in predictions]

        return predictions


    def sample_errors_p_e_given_s(self, batch):
        # Sample e hat from p(e|s) with batched inference
        self.tokenizer.padding_side = "left" # Batched inference requires tokenizer padding side as left
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        
        prompts = [create_prompt_e_given_s(item, self.configs, mode="without_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
        stop_token, stop_token_id = get_stop_token_info("e_given_s", self.tokenizer)
        with self.pipeline.model_ctm("e_given_s_adapter", "eval"):
            # Nucleus sampling with top-k
            outputs = self.pipeline.model.generate(
                **encodings,
                max_new_tokens = self.configs.max_new_tokens_error,
                do_sample = self.configs.do_sample_errors_train,
                top_p = self.configs.top_p,
                top_k = self.configs.top_k,
                eos_token_id = stop_token_id
                )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = post_process_predictions(predictions, prompts_len, stop_token, self.configs)

        return predictions


    def sample_distractors_d_given_s_e(self, batch):
        # Sample d from p(d|s,e) with batched inference
        self.tokenizer.padding_side = "left" # Batched inference requires tokenizer padding side as left
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        
        # Construct inputs for d_given_s_e
        prompts = [create_prompt_d_given_s_e(item, self.configs, mode="without_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
        stop_token, stop_token_id = get_stop_token_info("d_given_s_e", self.tokenizer)

        with self.pipeline.model_ctm("d_given_s_e_adapter", "eval"):
            # Greedy decoding
            outputs = self.pipeline.model.generate(
                **encodings,
                max_new_tokens = self.configs.max_new_tokens_distractor,
                eos_token_id = stop_token_id
                )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = post_process_predictions(predictions, prompts_len, stop_token, self.configs)

        return predictions


class CollateWrapperGenerativeWithoutErrorLabel(CollateWrapperParentWithoutErrorLabel):
    def __init__(self, tokenizer, device, configs, pipeline):
        super().__init__(tokenizer, device, configs, pipeline)

    def __call__(self, batch):
        # Monte Carlo sampling: Duplicate batch num Monte Carlo sample times, and reshuffle
        # For duplication use deep copy since elements are dictionaries
        monte_carlo_batch = []
        for _ in range(self.configs.num_monte_carlo_samples):
            monte_carlo_batch += copy.deepcopy(batch)
        batch = random.sample(monte_carlo_batch, len(monte_carlo_batch))

        # Sample e hat from q(e|s,d) with batched inference
        sampled_errors_q_e_given_s_d = self.sample_errors_q_e_given_s_d(batch)
        for index, item in enumerate(batch):
            item["predicted_error"] = sampled_errors_q_e_given_s_d[index]
        self.tokenizer.padding_side = "right" # Reset tokenizer padding side to right for creating inputs for training

        sampled_errors_p_e_given_s = ["-" for _ in range(len(batch))]
        sampled_distractors = ["-" for _ in range(len(batch))]
        sampled_errors_q_ref_e_given_s_d = ["-" for _ in range(len(batch))]
        if( self.configs.log_p_e_given_s ):
            # Sample e hat from p(e|s) with batched inference
            sampled_errors_p_e_given_s = self.sample_errors_p_e_given_s(batch)
            self.tokenizer.padding_side = "right" # Reset tokenizer padding side to right for creating inputs for training
        if( self.configs.log_d_given_s_e ):
            # Sample d from p(d|s,e) with batched inference
            sampled_distractors = self.sample_distractors_d_given_s_e(batch)
            self.tokenizer.padding_side = "right" # Reset tokenizer padding side to right for creating inputs for training
        if( self.configs.log_q_ref_e_given_s_d ):
            # Sample e hat from q_ref(e|s,d) with batched inference
            sampled_errors_q_ref_e_given_s_d = self.sample_errors_q_ref_e_given_s_d(batch)
            self.tokenizer.padding_side = "right" # Reset tokenizer padding side to right for creating inputs for training

        # Create inputs for p(e|s), q(e|s,d) and p(d|s,e)
        inputs = {}
        inputs["e_given_s"] = self.get_inputs(
            "e_given_s", batch, create_prompt_e_given_s, create_completion_e_given_s, get_error_mask=True, mode="without_error_label") 
        inputs["e_given_s_d"] = self.get_inputs(
            "e_given_s_d", batch, create_prompt_e_given_s_d, create_completion_e_given_s_d, get_error_mask=True, mode="without_error_label")
        inputs["d_given_s_e"] = self.get_inputs(
            "d_given_s_e", batch, create_prompt_d_given_s_e, create_completion_d_given_s_e, get_error_mask=True, mode="without_error_label")
        # For logging sampled errors
        inputs["question"] = [item["question"] for item in batch]
        inputs["option"] = [item["option"] for item in batch]
        inputs["misconception_name"] = [item["misconception_name"] for item in batch]
        inputs["sampled_error_q_e_given_s_d"] = sampled_errors_q_e_given_s_d
        inputs["sampled_error_p_e_given_s"] = sampled_errors_p_e_given_s
        inputs["sampled_distractor"] = sampled_distractors
        inputs["sampled_error_q_ref_e_given_s_d"] = sampled_errors_q_ref_e_given_s_d

        return inputs


class CollateWrapperGenerativeTestSampleErrors(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs, _pipeline):
        super().__init__(tokenizer, device, configs)


    def __call__(self, batch):
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        # Create inputs for p(e|s)
        prompts = [create_prompt_e_given_s(item, self.configs) for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)

        if( self.configs.debug ):
            print(f"prompts: {prompts}")
            print(f"prompts_len: {prompts_len}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            for ids in prompts_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))
                
        return {
            "inputs": prompts_tokenized.to(self.device),
            "prompts_len": prompts_len
        }


class CollateWrapperGenerativeTestSampleDistractors(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs, _pipeline):
        super().__init__(tokenizer, device, configs)


    def __call__(self, batch):
        #print(batch)
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        # Create inputs for p(d|s,e)
        prompts = [create_prompt_d_given_s_e(item, self.configs, mode="without_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
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


class CollateWrapperGenerativeTestSampleDistractorsGivenQuestionError(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs, _pipeline):
        super().__init__(tokenizer, device, configs)


    def __call__(self, batch):
        #print(batch)
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        # Create inputs for p(d|s,e)
        prompts = [create_prompt_d_given_s_e(item, self.configs, mode="with_error_label") for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
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
    

class CollateWrapperGenerativeTestSampleErrorsGivenQuestionDistractor(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs, _pipeline):
        super().__init__(tokenizer, device, configs)


    def __call__(self, batch):
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        # Create inputs for p(e|s,d)
        prompts = [create_prompt_e_given_s_d(item, self.configs) for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)

        if( self.configs.debug ):
            print(f"prompts: {prompts}")
            print(f"prompts_len: {prompts_len}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            for ids in prompts_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))

        return {
            "inputs": prompts_tokenized.to(self.device),
            "prompts_len": prompts_len
        }


class CollateWrapperGenerativeWithErrorLabel(CollateWrapperParent):
    def __init__(self, tokenizer, device, configs):
        super().__init__(tokenizer, device, configs)


    def __call__(self, batch):
        assert self.tokenizer.padding_side == "right" # Reset tokenizer padding side to right for creating inputs for training

        # Create inputs for p(e|s) and p(d|s,e)
        inputs = {}
        inputs["e_given_s"] = self.get_inputs("e_given_s", batch, create_prompt_e_given_s, create_completion_e_given_s, get_error_mask=False, mode="with_error_label") 
        inputs["d_given_s_e"] = self.get_inputs("d_given_s_e", batch, create_prompt_d_given_s_e, create_completion_d_given_s_e, get_error_mask=False, mode="with_error_label")

        return inputs