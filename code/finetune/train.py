import hydra
import wandb
import time
from omegaconf import OmegaConf
from tqdm import tqdm

from code.utils.utils import set_random_seed, aggregate_metrics, save_model, sanitize_configs, get_device, get_run_name, run_garbage_collector
from code.utils.load_data import load_data_finetune, get_data_loaders_finetune
from code.finetune.model import Trainer
from code.finetune.test_error_gen import test_error_gen
from code.finetune.test_distractor_gen import test_distractor_gen
from code.finetune.batch_collator import CollateWrapperGenerative


def train(configs, device):
    # Load model
    trainer = Trainer(configs, device)

    # Load data
    mode = "without_error_label" if configs.task_name == "d_given_s" else "with_error_label"
    train_set, val_set, test_set = load_data_finetune(configs, mode)
    train_loader, val_loader = get_data_loaders_finetune(train_set, val_set, CollateWrapperGenerative, trainer.tokenizer, device, configs)

    # Best validation metric
    best_val_metric = float("inf")
    best_val_loss_epoch = -1

    # Training loop
    with tqdm(range(configs.num_epochs)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Epoch {}".format(cur_iter))
            start_time = time.time()
            
            # Train epoch
            trainer.set_train_mode() # Set train mode for model
            train_logs = []
            with tqdm(train_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("Batch {}".format(batch_num))
                    logs = trainer.train_step(batch)  
                    train_logs.append(logs)
            
            # After every training epoch, push logs to weights and biases
            train_it_time = time.time() - start_time
            train_logs = aggregate_metrics(train_logs)
            if configs.log_wandb:
                wandb.log({"logs/train/it_time": train_it_time})
                wandb.log({"metrics/train/loss": train_logs['loss']})
                wandb.log({"logs/train/cur_iter" : cur_iter})

            run_garbage_collector()
            # Evaluate on validation set after every training epoch
            val_logs, best_val_metric, best_val_loss_epoch = validate(val_loader, best_val_metric, trainer, configs, cur_iter, best_val_loss_epoch)
            # Update training tqdm progress bar
            tepoch.set_postfix({"train loss" : train_logs['loss'], "val loss" : val_logs['loss']})
            run_garbage_collector()
    
    # Test with best saved model
    del trainer
    run_garbage_collector()

    return test_set


def validate(val_loader, best_val_metric, trainer, configs, cur_iter, best_val_loss_epoch):
    # Evaluation epoch
    # Set eval mode for model
    trainer.set_eval_mode()
    val_logs = []
    eval_start_time = time.time()
    for batch in val_loader:
        logs = trainer.val_step(batch)
        val_logs.append(logs)
    eval_it_time = time.time()-eval_start_time

    # Aggregate logs across batches
    val_logs = aggregate_metrics(val_logs)
    # Update metrics and save model
    if( float(val_logs["loss"]) < best_val_metric ):
        best_val_loss_epoch = cur_iter
        best_val_metric = float(val_logs["loss"])
        # Save model with best validation loss
        save_model(trainer, configs, get_run_name())
    # Push logs to weights and biases
    if configs.log_wandb:
        wandb.log({"logs/val/best_loss_epoch": best_val_loss_epoch})
        wandb.log({"metrics/val/loss": val_logs['loss']})
        wandb.log({"metrics/val/best_loss": best_val_metric})
        wandb.log({"logs/val/it_time": eval_it_time})
    
    return val_logs, best_val_metric, best_val_loss_epoch


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    # Get device
    device = get_device(configs)
    # Link training to weights and biases
    if configs.log_wandb:
        wandb.init(project=configs.wandb_project)
        wandb.config.update(OmegaConf.to_container(configs, resolve=True))
    # Train model
    test_set = train(configs, device)
    # Test with best saved model
    test = test_error_gen if ( configs.task_name == "e_given_s" or configs.task_name == "e_given_s_d" ) else test_distractor_gen
    test(test_set, get_run_name(), configs, device)


if __name__ == '__main__':
    main()