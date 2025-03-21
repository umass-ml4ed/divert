import hydra
import wandb
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import logging

from code.utils.utils import (set_random_seed, aggregate_metrics, save_model, sanitize_configs, get_device, save_sampled_errors,
    run_garbage_collector, is_batch_left, num_batches_left, get_run_name)
from code.utils.load_data import load_data_distractorgen, get_data_loaders_distractorgen
from code.distractorgen.model import Trainer
from code.distractorgen.test import test
from code.distractorgen.batch_collator import CollateWrapperGenerativeWithoutErrorLabel, CollateWrapperGenerativeWithErrorLabel


def train(configs, device):
    # Load model
    trainer = Trainer(configs, device)    
    train_set_without_error_label, train_set_with_error_label, val_set, test_set, _ = load_data_distractorgen(configs)
    collator_without_err_label = CollateWrapperGenerativeWithoutErrorLabel(trainer.tokenizer, device, configs, trainer.pipeline)
    collator_with_err_label = CollateWrapperGenerativeWithErrorLabel(trainer.tokenizer, device, configs)
    train_loader_without_error_label, train_loader_with_error_label, val_loader = get_data_loaders_distractorgen(train_set_without_error_label, train_set_with_error_label, 
                                                                                                                 val_set, collator_without_err_label,
                                                                                                                 collator_with_err_label, configs)

    # Best validation metric
    best_val_metric = float("inf")
    best_val_loss_epoch = -1
    best_val_d_given_s_e_loss = float("inf")
    best_val_d_given_s_e_loss_epoch = -1

    # Set exponential temperature decay rate so that target reached at last iteration
    num_its = configs.num_epochs * (len(train_loader_without_error_label) + len(train_loader_with_error_label))
    temperature_decay_rate = configs.softmax_temperature ** (1 / num_its)

    # Training loop
    with tqdm(range(configs.num_epochs)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Train epoch {}".format(cur_iter+1))
            start_time = time.time()
            trainer.set_train_mode() # Set train mode for model
            train_logs_without_error_label = []
            train_logs_with_error_label = []
            batches_sampled_error = []
            iter_train_loader_without_error_label = iter(train_loader_without_error_label)
            iter_train_loader_with_error_label = iter(train_loader_with_error_label)
            batch_num_without_error_label = 0
            batch_num_with_error_label = 0
            batch_num = 0
            batch_pbar = tqdm(total=(len(iter_train_loader_without_error_label) + len(iter_train_loader_with_error_label)), leave=False)

            while(is_batch_left(iter_train_loader_without_error_label) or is_batch_left(iter_train_loader_with_error_label)):
                # Train on batch without error label using gradient accumulation
                assert trainer.pipeline.train_with_error_label == False, "Error: train_with_error_label should be False"
                if is_batch_left(iter_train_loader_without_error_label):
                    # We normalize loss by grad_accum_steps: min ensures accurate normalization for the last few batches in the epoch < grad_accum_steps
                    grad_accum_steps = min(configs.num_grad_accum_steps, num_batches_left(iter_train_loader_without_error_label))
                    for _ in range(grad_accum_steps):
                        batch_without_error_label = next(iter_train_loader_without_error_label)
                        logs = trainer.loss_backward_step(batch_without_error_label, grad_accum_steps)
                        if configs.anneal_temperature:
                            trainer.pipeline.decay_softmax_temperature(temperature_decay_rate)
                            collator_without_err_label.decay_softmax_temperature(temperature_decay_rate)
                        train_logs_without_error_label.append(logs)
                        # Save sampled errors and distractors for batch without error label
                        batches_sampled_error.append({
                                        "batch_num": [batch_num_without_error_label for _ in range(len(batch_without_error_label["question"]))],
                                        "question": batch_without_error_label["question"], 
                                        "option": batch_without_error_label["option"], 
                                        "misconception_name": batch_without_error_label["misconception_name"],
                                        "sampled_error_q_e_given_s_d": batch_without_error_label["sampled_error_q_e_given_s_d"],
                                        "sampled_error_p_e_given_s": batch_without_error_label["sampled_error_p_e_given_s"],
                                        "sampled_distractor": batch_without_error_label["sampled_distractor"],
                                        "sampled_error_q_ref_e_given_s_d": batch_without_error_label["sampled_error_q_ref_e_given_s_d"],
                                        })
                        # Aggregate and log most recent k batches without error labels
                        if( (configs.log_k_batches) and ((batch_num_without_error_label+1) % configs.k_batches == 0) ) and configs.log_wandb and train_logs_without_error_label:
                            k_step_train_logs_without_error_label = aggregate_metrics(train_logs_without_error_label[-configs.k_batches::])
                            wandb.log({"metrics/train_k_batches/batch_num": batch_num_without_error_label})
                            wandb.log({"metrics/train_k_batches/loss": k_step_train_logs_without_error_label['loss']})
                            wandb.log({"metrics/train_k_batches/loss_e_given_s": k_step_train_logs_without_error_label['loss_e_given_s']})
                            wandb.log({"metrics/train_k_batches/loss_d_given_s_e": k_step_train_logs_without_error_label['loss_d_given_s_e']})
                            wandb.log({"metrics/train_k_batches/loss_e_given_s_d": k_step_train_logs_without_error_label['loss_e_given_s_d']})
                            if( configs.regularize_q ):
                                wandb.log({"metrics/train_k_batches/loss_regularize_q": k_step_train_logs_without_error_label['loss_regularize_q']})
                            if( configs.anneal_temperature ):
                                wandb.log({"metrics/train_k_batches/softmax_temperature": trainer.pipeline.softmax_temperature})
                        batch_num_without_error_label += 1
                        batch_num += 1
                        batch_pbar.update(1)
                        batch_pbar.set_description(f"Train batch {batch_num}")
                    # Update weights using gradient accumulation
                    trainer.grad_step()

                # Train on batch with error label using gradient accumulation
                if is_batch_left(iter_train_loader_with_error_label):
                    trainer.pipeline.train_with_error_label = True
                    grad_accum_steps = min(configs.num_grad_accum_steps, num_batches_left(iter_train_loader_with_error_label))
                    for _ in range(grad_accum_steps):
                        batch_with_error_label = next(iter_train_loader_with_error_label)
                        logs = trainer.loss_backward_step(batch_with_error_label, grad_accum_steps)
                        if configs.anneal_temperature:
                            trainer.pipeline.decay_softmax_temperature(temperature_decay_rate)
                            collator_without_err_label.decay_softmax_temperature(temperature_decay_rate)
                        train_logs_with_error_label.append(logs)
                        batch_num_with_error_label += 1
                        batch_num += 1
                        batch_pbar.update(1)
                        batch_pbar.set_description(f"Train batch {batch_num}")
                    # Update weights using gradient accumulation
                    trainer.grad_step()
                    trainer.pipeline.train_with_error_label = False

            # After every training epoch, push logs to weights and biases
            train_it_time = time.time() - start_time
            if train_logs_without_error_label:
                train_logs_without_error_label = aggregate_metrics(train_logs_without_error_label)
            if train_logs_with_error_label:
                train_logs_with_error_label = aggregate_metrics(train_logs_with_error_label)
            if configs.log_wandb:
                wandb.log({"logs/train/it_time": train_it_time})
                wandb.log({"logs/train/cur_iter" : cur_iter})
                # Train logs without error label
                if train_logs_without_error_label:
                    wandb.log({"metrics/train/loss": train_logs_without_error_label['loss']})
                    wandb.log({"metrics/train/loss_e_given_s": train_logs_without_error_label['loss_e_given_s']})
                    wandb.log({"metrics/train/loss_d_given_s_e": train_logs_without_error_label['loss_d_given_s_e']})
                    wandb.log({"metrics/train/loss_e_given_s_d": train_logs_without_error_label['loss_e_given_s_d']})
                    if( configs.regularize_q ):
                        wandb.log({"metrics/train/loss_regularize_q": train_logs_without_error_label['loss_regularize_q']})
                # Train logs with error label
                if train_logs_with_error_label:
                    wandb.log({"metrics/train/with_error_label/loss": train_logs_with_error_label['loss']})
                    wandb.log({"metrics/train/with_error_label/loss_e_given_s": train_logs_with_error_label['loss_e_given_s']})
                    wandb.log({"metrics/train/with_error_label/loss_d_given_s_e": train_logs_with_error_label['loss_d_given_s_e']})                              
            # Save sampled errors from q(e|d,s)
            if batches_sampled_error:
                save_sampled_errors(batches_sampled_error, cur_iter, configs, get_run_name())

            run_garbage_collector()
            # Evaluate on validation set after every training epoch
            val_logs, best_val_metric, best_val_loss_epoch, best_val_d_given_s_e_loss, best_val_d_given_s_e_loss_epoch = validate(val_loader, best_val_metric, trainer, configs, cur_iter, best_val_loss_epoch, best_val_d_given_s_e_loss, best_val_d_given_s_e_loss_epoch)
            # Update training tqdm progress bar
            train_loss = train_logs_without_error_label['loss'] if train_logs_without_error_label else train_logs_with_error_label['loss']
            tepoch.set_postfix({"Train loss" : train_loss, "Val loss" : val_logs['loss']})
            run_garbage_collector()
            batch_pbar.close()

    del trainer
    run_garbage_collector()

    return test_set


def validate(val_loader, best_val_metric, trainer, configs, cur_iter, best_val_loss_epoch, best_val_d_given_s_e_loss, best_val_d_given_s_e_loss_epoch):
    # Evaluation epoch
    configs.validation = True
    # Set eval mode for model
    trainer.set_eval_mode()
    val_logs = []
    eval_start_time = time.time()
    with tqdm(val_loader, leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Val batch {}".format(batch_num))
            logs = trainer.val_step(batch)
            val_logs.append(logs)
    eval_it_time = time.time()-eval_start_time

    # Aggregate logs across batches
    val_logs = aggregate_metrics(val_logs)
    # Update metrics and save model on best validation loss
    if( float(val_logs["loss"]) < best_val_metric ):
        best_val_loss_epoch = cur_iter
        best_val_metric = float(val_logs["loss"])
        save_model(trainer, configs, get_run_name(), "best_val_loss")
    # Update metrics and save model on best d_given_s_e validation loss
    if( float(val_logs["loss_d_given_s_e"]) < best_val_d_given_s_e_loss ):
        best_val_d_given_s_e_loss_epoch = cur_iter
        best_val_d_given_s_e_loss = float(val_logs["loss_d_given_s_e"])
        # Don't save model to save space for now
        #save_model(trainer, configs, get_run_name(), "best_val_d_given_s_e_loss")
    # Push logs to weights and biases
    if configs.log_wandb:
        wandb.log({"logs/val/best_loss_epoch": best_val_loss_epoch})
        wandb.log({"metrics/val/best_loss": best_val_metric})
        wandb.log({"logs/val/best_d_given_s_e_loss_epoch": best_val_d_given_s_e_loss_epoch})
        wandb.log({"metrics/val/best_d_given_s_e_loss": best_val_d_given_s_e_loss})
        wandb.log({"logs/val/it_time": eval_it_time})
        wandb.log({"metrics/val/loss": val_logs['loss']})
        wandb.log({"metrics/val/loss_e_given_s": val_logs['loss_e_given_s']})
        wandb.log({"metrics/val/loss_d_given_s_e": val_logs['loss_d_given_s_e']})
        wandb.log({"metrics/val/loss_e_given_s_d": val_logs['loss_e_given_s_d']})
        if( configs.regularize_q ):
            wandb.log({"metrics/val/loss_regularize_q": val_logs['loss_regularize_q']})
    configs.validation = False

    return val_logs, best_val_metric, best_val_loss_epoch, best_val_d_given_s_e_loss, best_val_d_given_s_e_loss_epoch


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Turn off warnings
    logging.set_verbosity_error()
    # Make reproducible
    set_random_seed(configs.seed)
    configs = sanitize_configs(configs)
    device = get_device(configs) 
    if configs.log_wandb:
        wandb.init(project=configs.wandb_project)
        wandb.config.update(OmegaConf.to_container(configs, resolve=True))
    test_set = train(configs, device)
    # Test with best saved model
    test(test_set, get_run_name(), configs, device)


if __name__ == '__main__':
    main()
