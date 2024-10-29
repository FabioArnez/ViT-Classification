import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from datasets import CIFAR10DataModule
from models import ViTConfigExtended
from models import VisionTransformerModule
import hydra
from omegaconf import DictConfig
from helper_functions import log_params_from_omegaconf_dict
from datetime import datetime
from icecream import ic
import lightning as L 


@hydra.main(version_base=None, config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.ind_dataset in ("gtsrb", "cifar10", "cifar100")
    assert cfg.ind_dataset in cfg.data_dir
    #####################
    #      Get Args     #
    #####################
    model_type = cfg.model.model_type
    max_nro_epochs = cfg.trainer.epochs
    batch_size = cfg.datamodule.batch_size
    random_seed_everything = cfg.seed
    dataset_path = cfg.data_dir
    loss_type = cfg.model.loss_fn
    rich_progbar = cfg.rich_progbar
    slurm_training = cfg.slurm
    accelerator_device = cfg.trainer.accelerator
    devices = cfg.trainer.devices
    # Get current date time to synchronize pl logs and mlflow
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(' ')
    print('=' * 60)
    ic(current_date_time)
    ic(model_type)
    ic(max_nro_epochs)
    ic(batch_size)
    ic(loss_type)
    ic(random_seed_everything)
    ic(slurm_training)
    ic(accelerator_device)
    ic(devices)
    print('=' * 60)
    print(' ')

    ############################
    #      Seed Everything     #
    ############################
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pl.seed_everything(random_seed_everything)
    #######################################
    #      Training Monitor/Callbacks     #
    #######################################
    checkpoint_callback = ModelCheckpoint(dirpath=f'lightning_logs/{current_date_time}_{cfg.ind_dataset}',
                                          monitor=cfg.callbacks.model_checkpoint.monitor,
                                          mode=cfg.callbacks.model_checkpoint.mode,
                                          every_n_epochs=cfg.callbacks.model_checkpoint.every_n_epochs,
                                          save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
                                          save_last=cfg.callbacks.model_checkpoint.save_last,
                                          save_on_train_epoch_end=cfg.callbacks.model_checkpoint.save_on_train_epoch_end)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if rich_progbar:  # fancy aesthetic progress bar
        progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                                  progress_bar="green1",
                                                                  progress_bar_finished="green1",
                                                                  batch_progress="green_yellow",
                                                                  time="grey82",
                                                                  processing_speed="grey82",
                                                                  metrics="grey82"))
    else:  # normal aesthetic progress bar
        progress_bar = TQDMProgressBar(refresh_rate=cfg.trainer.progress_bar_refresh_rate)

    
    ###############################
    #      Get Dataset Module     #
    ###############################
    data_module = CIFAR10DataModule(data_dir=cfg.data_dir,
                                    img_size=(cfg.datamodule.image_width,
                                              cfg.datamodule.image_height),
                                    batch_size=cfg.datamodule.batch_size,
                                    valid_size=cfg.datamodule.valid_size,
                                    seed=cfg.datamodule.seed,
                                    transforms_type=cfg.datamodule.transforms_type,
                                    num_workers=cfg.datamodule.num_workers)
    num_classes = data_module.num_classes


    #############################
    #      Get Model Module     #
    #############################
    model_module = VisionTransformerModule(config=cfg)

    ########################################
    #      Start Module/Model Training     #
    ########################################
    mlf_logger = MLFlowLogger(experiment_name=cfg.logger.mlflow.experiment_name,
                              run_name=current_date_time,
                              tracking_uri=cfg.logger.mlflow.server_uri)

    model_trainer = pl.Trainer(logger=mlf_logger,
                               accelerator=cfg.trainer.accelerator,
                               devices=cfg.trainer.devices,
                               max_epochs=cfg.trainer.epochs,
                               callbacks=[progress_bar,
                                          lr_monitor,
                                          checkpoint_callback])
    # Log parameters with mlflow
    log_params_from_omegaconf_dict(cfg)
    # Setup automatic logging of training with mlflow
    mlflow.pytorch.autolog(checkpoint_monitor=cfg.callbacks.model_checkpoint.monitor)

    # Fit Trainer
    model_trainer.fit(model=model_module, datamodule=data_module)  # fit a model!


if __name__ == "__main__":
    main()
