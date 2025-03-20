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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Parameters import *
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.augmentation.ocrodeg import OcrodegAug


from torch.utils.data import DataLoader
from src.data.utils.custom_collate import custom_collate
from src.model.modules.WriterSequence import WriterSequence
from src.utils.utils import *
from argparse import ArgumentParser


#from pytorch_lightning.profiler import SimpleProfiler
from src.data.augmentation.noTransform import NoTransform
def writer_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--WriterConfigFile', type=str,default="768x768WriterCNN.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="general768x768.yaml") #TODO is this the correct DL?
    parser.add_argument('--reset_optimizers_writer', action='store_true',default=False)
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)

    return parser.parse_args()

if __name__ == "__main__":

    cfg = writer_parse_args()

    augment = OcrodegAug(p_dilation=0.3, p_erosion=0.3, p_distort_with_noise=0.3, p_elastic_distortion=0.3,
                         p_random_transform=0.3, p_perspective=0.3)

    gdm = instantiate_completely(r"Dataloaders/768x768", cfg.DataloaderConfigFile, val_only_real= False,

                                 batch_size=cfg.batch_size,augmentation=augment)
    logger = TensorBoardLogger(save_dir="TensorLogs/SeqWriterPara",name=cfg.name)

    # instantiating model
    model = instantiate_completely(STYLE_APLLY, cfg.WriterConfigFile)
    args = OmegaConf.load(get_yaml(STYLE_APLLY, cfg.WriterConfigFile))

    if args.get("ckpt") is not None and not cfg.reset_optimizers_writer:
        checkpoint_path = args["ckpt"]
    else:
        checkpoint_path = None

    es = EarlyStopping(monitor="val/acc", patience=50, mode="max")
    mc = ModelCheckpoint(save_top_k=3, monitor="val/acc", mode="max",
                         filename='{epoch}-{val/acc:.4f}')

    trainer = pl.Trainer(accelerator="gpu",devices=gpu_count,logger=logger,
                         callbacks=[es,mc],accumulate_grad_batches=cfg.accumulate_grad_batches)
    trainer.fit(model, train_dataloaders=gdm.train_dataloader(),val_dataloaders=gdm.val_dataloader(),
                ckpt_path=checkpoint_path)


