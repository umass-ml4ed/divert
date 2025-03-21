import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


class LanguageModel(nn.Module):
    def __init__(self, configs, device, mode, wandb_run_name=None):
        super().__init__()
        self.configs = configs
        self.device = device
        self.wandb_run_name = wandb_run_name
        if( mode == "train" ):
            self.init_train()
        else:
            self.wandb_run_name = wandb_run_name
            self.init_test()

    def init_train(self):
        self._bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
            )

        self._peft_config = LoraConfig(
            lora_alpha=self.configs.lora_alpha,
            lora_dropout=self.configs.lora_dropout,
            r=self.configs.lora_r,
            bias="none",
            task_type="CAUSAL_LM", 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            inference_mode=False
            )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.configs.model_name,
            quantization_config=self._bnb_config,
            device_map="auto"
            )
        self._hf_model.config.pretraining_tp = 1 
        self._hf_model = prepare_model_for_kbit_training(self._hf_model)
        self.model = get_peft_model(self._hf_model, self._peft_config)


    def init_test(self):
        # Quantize to match evaluation setting for distractor gen variational pipeline

        if( self.configs.zero_shot ):
            self._bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.configs.model_name,
                quantization_config=self._bnb_config,
                device_map="auto"
                )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                )   
            model_dir = f"{self.configs.model_checkpoint_dir}/{self.configs.task_name}/{self.wandb_run_name}/best_val_loss/lora_model"
            peft_config = PeftConfig.from_pretrained(model_dir)
            _hf_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto"
                )
            self.model = PeftModel.from_pretrained(_hf_model, model_dir, is_trainable=False).to(self.device)
        # Set model to eval mode
        self.model.eval()


    def forward(self, **kwargs):
        outputs = self.model(**kwargs)

        return outputs.loss


class Trainer(): 
    def __init__(self, configs, device):
        self.configs = configs
        self.model = LanguageModel(configs, device, "train").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.optimizer = AdamW(self.model.parameters(), lr=configs.lr)
        

    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self):
        # Gradient clipping
        if( self.configs.use_grad_clip ):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.grad_clip)
        self.optimizer.step()


    def train_step(self, batch):
        self.zero_grad()
        loss = self.model(**batch)
        loss.backward()
        self.grad_step()

        return {
            'loss': loss.detach().cpu()
            }


    def val_step(self, batch):
        with torch.no_grad():
            loss = self.model(**batch)

        return {
            'loss': loss.detach().cpu()
            }


    def set_train_mode(self):
        # Recursively sets children to train mode, therefore model.model.train() is also set
        self.model.train()


    def set_eval_mode(self):
        # Recursively sets children to eval mode, therefore model.model.eval() is also set
        self.model.eval()