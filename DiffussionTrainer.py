import string
import sys

import torch
from torch import nn
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl

from src.data.augmentation.noTransform import NoTransform
from src.data.dataset_fetcher import fetch_dataset
from src.data.utils.alphabet import Alphabet
from src.data.utils.constants import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Parameters import *
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.augmentation.ocrodeg import OcrodegAug
from torch.utils.data import DataLoader
from src.data.utils.custom_collate import custom_collate
from src.diffusion.ddpm import DDPM
from src.diffusion.ddpm import LatentDiffusion
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser


from src.utils.utils import *

def diffusion_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--DiffusionConfigFile', type=str,default="ours.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="general768x768.yaml") #TODO is this the correct DL?
    parser.add_argument('--reset_optimizers_ldm', action='store_true',default=False)
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--strategy', type=str, default="demo")
    parser.add_argument('--finetuning', action='store_true',default=True)

    return parser.parse_args()

if __name__ == "__main__":
    cfg = diffusion_parse_args()

    if cfg.finetuning:
        gdm = instantiate_completely(r"Diffusion/Dataloaders/finetune", cfg.DataloaderConfigFile)
        max_steps = 8000
    else:
        gdm = instantiate_completely(r"Diffusion/Dataloaders/general", cfg.DataloaderConfigFile)
        max_steps = 70000

    # instantiating model + logger
    logger = TensorBoardLogger(save_dir="TensorLogs/Diffusion",name=cfg.name)

    if cfg.strategy == "demo":
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger,
                             callbacks=[ModelCheckpoint()], max_steps=max_steps)
    else:

        trainer = pl.Trainer(accelerator="gpu", devices=8, logger=logger,
                                  accumulate_grad_batches=accumulate_grad_batches,
                                 strategy=DDPStrategy(find_unused_parameters=True),callbacks=[ModelCheckpoint()],
                             max_steps=max_steps)

    model = instantiate_completely("Diffusion/ldm/", cfg.DiffusionConfigFile)

    if cfg.reset_optimizers_ldm:
        checkpoint_path_etc = None
    else:
        checkpoint_path_etc = ld_ckpt_path

    trainer.fit(model, train_dataloaders=gdm.train_dataloader(), val_dataloaders=gdm.val_dataloader(),
                ckpt_path=checkpoint_path_etc)



