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
from thirdparty.VQVAEGAN.autoencoder import AutoencoderKL
from Parameters import *

from src.data.augmentation.ocrodeg import OcrodegAug
from torch.utils.data import DataLoader
from src.data.utils.custom_collate import custom_collate
from src.model.modules.HTR_Writer import HTR_Writer
from src.model.modules.WriterSequence import WriterSequence
from src.utils.utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from pytorch_lightning.strategies.ddp import DDPStrategy


def autoKL_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--AutoKLConfigFile', type=str,default="768x768AutoKL.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="general768x768.yaml") #TODO is this the correct DL?
    parser.add_argument('--reset_optimizers_auto', action='store_true',default=False)
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    parser.add_argument('--strategy', type=str, default="demo")

    return parser.parse_args()

"""
    If you plan to train this yourself take into account that it can take multiple days 
"""

if __name__ == "__main__":

    cfg = autoKL_parse_args()

    augment = OcrodegAug(p_dilation=0.3, p_erosion=0.3, p_distort_with_noise=0.3)


    gdm = instantiate_completely("Dataloaders/768x768", cfg.DataloaderConfigFile,
                                 augmentation= augment,batch_size=cfg.batch_size)
    logger = TensorBoardLogger(save_dir="TensorLogs/AutoEncoder",name=cfg.name)

    #intantiating Model
    model = instantiate_completely("AutoKL", cfg.AutoKLConfigFile)
    args = OmegaConf.load(get_yaml("AutoKL", cfg.AutoKLConfigFile))

    # Checkpoint?
    if (cfg.reset_optimizers_auto == False) and args.get("ckpt") is not None:
        checkpoint_path = args["ckpt"]
    else:
        checkpoint_path = None

    cb = EarlyStopping(monitor="val/rec_loss", mode="min", patience=30)
    mc = ModelCheckpoint(save_top_k=3, monitor="val/rec_loss", mode="min",
                         filename='{epoch}-{val/rec_loss:.4f}')

    if "demo" == cfg.strategy:
        trainer = pl.Trainer( accelerator="gpu", devices=1, logger=logger, callbacks=[cb,mc],
                              accumulate_grad_batches=cfg.accumulate_grad_batches)
    else:
        #in case you wanna train it on multi gpu
        trainer = pl.Trainer( accelerator="gpu", devices=2, logger=logger,callbacks=[cb,mc],
                             accumulate_grad_batches=cfg.accumulate_grad_batches,
                              strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model, train_dataloaders=gdm.train_dataloader(), val_dataloaders=gdm.val_dataloader(),
                ckpt_path=checkpoint_path)



