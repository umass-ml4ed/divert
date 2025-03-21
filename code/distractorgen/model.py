import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager

from code.utils.utils import get_gpu_memory, get_adapter_dirs


class DistractorGenModel(nn.Module):
    def __init__(self, configs, device, mode, wandb_run_name=None, use_base_models=False):
        super().__init__()
        self.configs = configs
        self.device = device
        if( mode == "train" ):
            self.init_train()
        else:
            self.wandb_run_name = wandb_run_name
            # Use base models before VAE training for performance ablation
            self.use_base_models = use_base_models
            self.init_test()


    def init_train(self):
        self._bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
            )

        # Load pretrained q(e|s,d), p(e|s), and p(d|e,s) models
        self.adapter_names = ["e_given_s_d_adapter", "e_given_s_adapter", "d_given_s_e_adapter"]
        self.adapter_frozen = {
            "e_given_s_d_adapter": self.configs.freeze_q,
            "e_given_s_adapter": self.configs.freeze_prior,
            "d_given_s_e_adapter": self.configs.freeze_likelihood,
            "e_given_s_d_ref_adapter": True
        }
        self.adapter_dirs = get_adapter_dirs(self.configs)
        _peft_config = PeftConfig.from_pretrained(self.adapter_dirs["e_given_s_d_ref_adapter"]) # Assumes base HF model is same for all PEFTs models

        _hf_model = AutoModelForCausalLM.from_pretrained(
            _peft_config.base_model_name_or_path,
            quantization_config=self._bnb_config,
            device_map="auto"
            )
        _hf_model.config.pretraining_tp = 1
        _hf_model = prepare_model_for_kbit_training(_hf_model)

        # Load PEFT model with e_given_s_d_ref adapter. Since e_given_s_d_ref is frozen, set is_trainable to False
        print(f"Loading adapter: e_given_s_d_ref")
        get_gpu_memory()
        self.model = PeftModel.from_pretrained(_hf_model, self.adapter_dirs["e_given_s_d_ref_adapter"], adapter_name=f"e_given_s_d_ref_adapter", is_trainable=False).to(self.device)
        for adapter_name in self.adapter_names:
            print(f"Loading adapter: {adapter_name}")
            self.model.load_adapter(self.adapter_dirs[adapter_name], adapter_name, is_trainable=not self.adapter_frozen[adapter_name])
            get_gpu_memory()
        self.model.train()
        # Flag set true if train batch has ground truth error labels else false if using sampled errors from q(e|s,d)
        self.train_with_error_label = False
        self.softmax_temperature = 1.0 if self.configs.anneal_temperature else self.configs.softmax_temperature


    def init_test(self):
        # Load models
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            )   
        if ( self.use_base_models ):
            self.adapter_dirs = get_adapter_dirs(self.configs)
            adapter_dir = self.adapter_dirs["d_given_s_e_adapter"]
        else:
            adapter_dir = f"{self.configs.model_checkpoint_dir}/{self.wandb_run_name}/d_given_s_e/best_val_loss/lora_model/d_given_s_e_adapter"
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto"
            )
        # Load PEFT model with d_given_s_e adapter
        print(f"Loading adapter: d_given_s_e from dir {adapter_dir}")
        self.model = PeftModel.from_pretrained(_hf_model, adapter_dir, adapter_name=f"d_given_s_e_adapter", is_trainable=False).to(self.device)
        get_gpu_memory()
        # Load remaining adapters
        adapter_names = ["e_given_s_d_adapter", "e_given_s_adapter"]
        for adapter_name in adapter_names:
            if( self.use_base_models ):
                adapter_dir = self.adapter_dirs[adapter_name]
            else:
                adapter_dir = f"{self.configs.model_checkpoint_dir}/{self.wandb_run_name}/{adapter_name.split('_adapter')[0]}/best_val_loss/lora_model/{adapter_name}"
            print(f"Loading adapter: {adapter_name} from dir {adapter_dir}")
            self.model.load_adapter(adapter_dir, adapter_name, is_trainable=False)
            get_gpu_memory()
        self.model.eval()
        # Needed since referenced in model_ctm
        self.adapter_frozen = {
            "e_given_s_d_adapter": self.configs.freeze_q,
            "e_given_s_adapter": self.configs.freeze_prior,
            "d_given_s_e_adapter": self.configs.freeze_likelihood,
            "e_given_s_d_ref_adapter": True
        }


    def forward(self, batch):
        if( self.train_with_error_label ):
            # Compute loss p(e|s)
            with self.model_ctm("e_given_s_adapter", "train"):
                outputs_e_given_s = self.model(input_ids=batch["e_given_s"]["input_ids"], attention_mask=batch["e_given_s"]["attention_mask"], labels=batch["e_given_s"]["labels"])
            loss_e_given_s = outputs_e_given_s.loss

            # Compute loss p(d|s,e)
            with self.model_ctm("d_given_s_e_adapter", "train"):
                outputs_d_given_s_e = self.model(input_ids=batch["d_given_s_e"]["input_ids"], attention_mask=batch["d_given_s_e"]["attention_mask"], labels=batch["d_given_s_e"]["labels"])
            loss_d_given_s_e = outputs_d_given_s_e.loss

            # No entropy term in loss
            if( self.configs.use_beta_kl ):
                loss = loss_d_given_s_e + torch.mul(self.configs.beta_kl, loss_e_given_s)
            else:
                loss = torch.mul(self.configs.weight_loss_d_given_s_e, loss_d_given_s_e) + torch.mul(self.configs.weight_loss_e_given_s, loss_e_given_s)
            loss_dict = {
                "loss": loss,
                "loss_d_given_s_e": loss_d_given_s_e,
                "loss_e_given_s": loss_e_given_s,
            }

        else:
            # Compute loss q(e|s,d)
            if not self.configs.freeze_q:
                with self.model_ctm("e_given_s_d_adapter", "train"):
                    outputs_e_given_s_d = self.model(input_ids=batch["e_given_s_d"]["input_ids"], attention_mask=batch["e_given_s_d"]["attention_mask"], labels=batch["e_given_s_d"]["labels"])
                    word_embeddings = self.model.get_input_embeddings()
                loss_e_given_s_d = outputs_e_given_s_d.loss
                logits = outputs_e_given_s_d.logits # Shape (B, S, V)
                soft_tokens = self.get_soft_error_tokens(word_embeddings, logits, batch["e_given_s_d"]["input_ids"]) # Shape (B, S, D)
            else:
                loss_e_given_s_d = torch.tensor(0).to(self.device)

            # Compute loss p(e|s)
            if self.configs.freeze_q:
                model_inputs = {"input_ids": batch["e_given_s"]["input_ids"]}
            else:
                batch["e_given_s"]["inputs_embeds"] = self.insert_soft_error_tokens(soft_tokens, batch["e_given_s"], batch["e_given_s"]["error_mask"], batch["e_given_s_d"]["error_mask"], word_embeddings)
                model_inputs = {"inputs_embeds": batch["e_given_s"]["inputs_embeds"]}
            with self.model_ctm("e_given_s_adapter", "train"):
                outputs_e_given_s = self.model(**model_inputs, attention_mask=batch["e_given_s"]["attention_mask"], labels=batch["e_given_s"]["labels"])
            loss_e_given_s = outputs_e_given_s.loss

            # Compute loss p(d|s,e)
            if self.configs.freeze_q:
                model_inputs = {"input_ids": batch["d_given_s_e"]["input_ids"]}
            else:
                batch["d_given_s_e"]["inputs_embeds"] = self.insert_soft_error_tokens(soft_tokens, batch["d_given_s_e"], batch["d_given_s_e"]["error_mask"], batch["e_given_s_d"]["error_mask"], word_embeddings)
                model_inputs = {"inputs_embeds": batch["d_given_s_e"]["inputs_embeds"]}
            with self.model_ctm("d_given_s_e_adapter", "train"):
                outputs_d_given_s_e = self.model(**model_inputs, attention_mask=batch["d_given_s_e"]["attention_mask"], labels=batch["d_given_s_e"]["labels"])
            loss_d_given_s_e = outputs_d_given_s_e.loss

            if( self.configs.regularize_q ):
                # Compute regularization loss between current q model and reference q model for q(e|s,d)
                with self.model_ctm("e_given_s_d_ref_adapter", "eval"):
                    with torch.no_grad():
                        outputs_e_given_s_d_ref = self.model(input_ids=batch["e_given_s_d"]["input_ids"], attention_mask=batch["e_given_s_d"]["attention_mask"], labels=batch["e_given_s_d"]["labels"])
                # Targets for cross entropy loss need to be normalized between [0,1]
                targets_ref = F.softmax(outputs_e_given_s_d_ref.logits, dim=-1)
                inputs = outputs_e_given_s_d.logits
                cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
                loss_regularize_q = cross_entropy_loss(inputs.view(-1, inputs.shape[-1]), targets_ref.view(-1, targets_ref.shape[-1]))
                # Compute cross entropy loss over error tokens only
                mask = batch["e_given_s_d"]["error_mask"]
                # Shift mask left by 1 since logits are for next token prediction (logit i corresponds to token i+1)
                mask_left_shifted = torch.cat([mask[:, 1:], torch.zeros(mask.shape[0], 1).to(self.device)], dim=1).bool()
                loss_regularize_q = (loss_regularize_q * mask_left_shifted.float().view(-1)).sum()
                num_non_zero_elements = mask_left_shifted.sum()
                epsilon = 1e-7
                loss_regularize_q = torch.div(loss_regularize_q, num_non_zero_elements+epsilon)

            if( self.configs.use_beta_kl ):
                loss = loss_d_given_s_e - torch.mul(self.configs.beta_kl, loss_e_given_s_d) + torch.mul(self.configs.beta_kl, loss_e_given_s) 
            else:
                loss = torch.mul(self.configs.weight_loss_d_given_s_e, loss_d_given_s_e) - torch.mul(self.configs.weight_loss_e_given_s_d, loss_e_given_s_d) + torch.mul(self.configs.weight_loss_e_given_s, loss_e_given_s)

            loss_dict = {
                "loss_d_given_s_e": loss_d_given_s_e,
                "loss_e_given_s": loss_e_given_s,
                "loss_e_given_s_d": loss_e_given_s_d,
            }
            if( self.configs.regularize_q ):
                loss = loss + torch.mul(self.configs.weight_loss_regularize_q, loss_regularize_q)
                loss_dict["loss_regularize_q"] = loss_regularize_q
            loss_dict["loss"] = loss

        return loss_dict


    def decay_softmax_temperature(self, decay_rate):
        self.softmax_temperature *= decay_rate


    def get_soft_error_tokens(self, word_embeddings: torch.Tensor, logits: torch.Tensor, input_tokens: torch.Tensor):
        # Apply peaked softmax
        vocab_distro = F.softmax(torch.divide(logits, self.softmax_temperature), dim=-1) # Shape (B, S, V)

        # Convert distribution to embeddings
        soft_tokens = torch.matmul(vocab_distro, word_embeddings.weight) # Shape (B, S, D) = (B, S, V) X (V, D)

        return soft_tokens


    def insert_soft_error_tokens(self, soft_tokens, inputs, target_mask, source_mask, word_embeddings):
        # Shift source mask left by 1 since logits used in soft error tokens are for next token prediction (logit i corresponds to token i+1)
        source_mask_left_shifted = torch.cat([source_mask[:, 1:], torch.zeros(source_mask.shape[0], 1).to(self.device)], dim=1).bool()
        # Check number of soft error tokens expected in target is equal to number of soft error tokens in source
        assert torch.sum(target_mask) == torch.sum(source_mask_left_shifted), "Error: Target and source masks are incompatible"
        # Convert input ids to input embeddings
        inputs_embeds = word_embeddings(inputs["input_ids"]) # Shape: (B, S, D)
        # Get soft error tokens
        soft_error_tokens = soft_tokens[source_mask_left_shifted] # Shape: (sum(S==true), D)
        # Insert soft error tokens to target
        inputs_embeds = torch.masked_scatter(inputs_embeds, target_mask.unsqueeze(dim=2), soft_error_tokens) # Shape: (B, S, D)

        return inputs_embeds


    @contextmanager
    def model_ctm(self, adapter_name, mode):
        if( isinstance(adapter_name, str) ):
            self.model.set_adapter(adapter_name)
            frozen = self.adapter_frozen[adapter_name]
        else:
            # Set multiple adapters at once: adapter_name is a list of strings
            # https://github.com/huggingface/peft/discussions/1315
            self.model.base_model.set_adapter(adapter_name)
            frozen = False
        if( mode == "train" and not frozen ):
            self.model.train()
        else:
            self.model.eval()
        yield
        # Reset back to default adapter and train mode
        self.model.set_adapter("d_given_s_e_adapter")
        self.model.train()


class Trainer():
    def __init__(self, configs, device):
        self.configs = configs
        self.device = device
        self.pipeline = DistractorGenModel(configs, device, "train").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pipeline.adapter_dirs["e_given_s_d_adapter"])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.optimizer = AdamW(self.pipeline.model.parameters(), lr=configs.lr)


    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self):
        # Gradient clipping
        if( self.configs.use_grad_clip ):
            nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), self.configs.grad_clip)
        self.optimizer.step()
        self.zero_grad()


    def loss_backward_step(self, batch, grad_accum_steps):
        loss_dict = self.pipeline(batch)
        loss = loss_dict["loss"]
        loss_dict = {k: v.item() for (k, v) in loss_dict.items()}

        adapter_names = self.pipeline.adapter_names if not self.pipeline.train_with_error_label else ["e_given_s_adapter", "d_given_s_e_adapter"]
        adapter_names = [adapter_name for adapter_name in adapter_names if not self.pipeline.adapter_frozen[adapter_name]]
        # Set all adapters as active in train mode to compute gradients in pipeline
        with self.pipeline.model_ctm(adapter_names, "train"):
            # Normalize loss by number of gradient accumulation steps (although we log the original unnormalized loss)
            loss = torch.div(loss, grad_accum_steps)
            loss.backward()

        return loss_dict


    def val_step(self, batch):
        with torch.no_grad():
            loss_dict = self.pipeline(batch)

        return {k: v.item() for (k, v) in loss_dict.items()}


    def set_train_mode(self):
        # Recursively sets children to train mode, therefore model.model.train() is also set
        self.pipeline.model.train()


    def set_eval_mode(self):
        # Recursively sets children to eval mode, therefore model.model.eval() is also set
        self.pipeline.model.eval()