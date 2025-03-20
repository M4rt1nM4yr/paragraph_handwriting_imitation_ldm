import string
import sys

import torch
from torch import nn
from torchvision import transforms
import pytorch_lightning as pl

from src.data.augmentation.noTransform import NoTransform
from src.data.dataset_fetcher import fetch_dataset
from src.data.utils.alphabet import Alphabet
from src.data.utils.constants import *
from src.model.modules.HTR_Writer import HTR_Writer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Parameters import *

from src.data.augmentation.ocrodeg import OcrodegAug
from pytorch_lightning.callbacks import ModelCheckpoint


from torch.utils.data import DataLoader
from src.data.utils.custom_collate import custom_collate
from src.utils.utils import *
from argparse import ArgumentParser


from src.data.augmentation.noTransform import NoTransform



def htr_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--HTRConfigFile', type=str,default="HTR768x768.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="general768x768.yaml")
    parser.add_argument('--reset_optimizers_htr', action='store_true',default=False)
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__":

    cfg = htr_parse_args()
    augment = OcrodegAug(p_dilation=0.3, p_erosion=0.3, p_distort_with_noise=0.3, p_elastic_distortion=0.3,
                         p_random_transform=0.3, p_perspective=0.3)

    logger = TensorBoardLogger(save_dir="TensorLogs/HTR",name=cfg.name)
    gdm = instantiate_completely("Dataloaders/768x768", cfg.DataloaderConfigFile,
                                 augmentation = augment,batch_size=cfg.batch_size)

    model = instantiate_completely("HTR", cfg.HTRConfigFile)
    args = OmegaConf.load(get_yaml("HTR", cfg.HTRConfigFile))

    checkpoint_path = None
    if args.get("ckpt") is not None and not cfg.reset_optimizers_htr:
        checkpoint_path = args["ckpt"]
    else:
        checkpoint_path = None

    mc = ModelCheckpoint(save_top_k=3, monitor="val/cer",mode="min",filename='{epoch}-{val/cer:.4f}')
    es = EarlyStopping(monitor="val/cer", patience=50, mode="min")

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=gpu_count, logger=logger,
                             callbacks=[es,mc], accumulate_grad_batches=cfg.accumulate_grad_batches)

    trainer.fit(model, train_dataloaders=gdm.train_dataloader(),val_dataloaders=gdm.val_dataloader(),
                ckpt_path=checkpoint_path)


