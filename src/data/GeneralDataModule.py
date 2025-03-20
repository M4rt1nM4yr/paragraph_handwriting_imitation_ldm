import os.path
from typing import Optional

import pandas.core.computation.parsing
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

import Parameters
from src.data.augmentation.noTransform import NoTransform
# from src.data.IAMDataset import IAMDataset
from src.data.utils.custom_collate import custom_collate
from src.data.GeneralDataset import AbstractDataset
from src.utils.utils import *

from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class GeneralDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset=AbstractDataset,
                 dataset_type = "combined",
                 dataset_type_val = "IAM",
                 dataset_type_test = "IAM",
                 augmentation: nn.Module=NoTransform(),
                 batch_size=1, n_workers=8,
                 stage = "train",
                 persistent_workers = True,
                 train_h5=None,
                 val_h5=None,
                 test_h5=None,
                 scale=1.0,
                 style_samples=False,
                 load_test = True,
                 **kwargs):
        super(GeneralDataModule, self).__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.persistent_workers = persistent_workers

        if isinstance(dataset,str):
            dataset = get_obj_from_str(dataset)

        if stage != "test":
            self.train_data: AbstractDataset = dataset(split="train",dataset_file=train_h5,
                                                       augmentation=augmentation,style_samples=style_samples,
                                                       scale=scale,dataset_type=dataset_type,**kwargs)

            self.val_data: AbstractDataset = dataset(split="val",dataset_file=val_h5,style_samples=style_samples,
                                                     scale=scale,dataset_type=dataset_type_val,**kwargs)
            assert isinstance(self.train_data, AbstractDataset)
            assert isinstance(self.val_data, AbstractDataset)
            log.info(f"Train set size: {len(self.train_data)} | Validation set size: {len(self.val_data)}")

        if load_test:
            self.test_data: AbstractDataset = dataset(split="test",dataset_file=test_h5,style_samples=style_samples,
                                                      dataset_type=dataset_type_test,scale=scale,**kwargs)
            assert isinstance(self.test_data, AbstractDataset)
            log.info(f"Test set size: {len(self.test_data)}")

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, collate_fn=custom_collate, batch_size=self.batch_size,
                          num_workers=self.n_workers, shuffle=True,persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, collate_fn=custom_collate, batch_size=self.batch_size,
                          num_workers=self.n_workers,persistent_workers=self.persistent_workers)
    def test_dataloader(self):
        return DataLoader(self.test_data, collate_fn=custom_collate, batch_size=self.batch_size,
                          num_workers=self.n_workers,persistent_workers=self.persistent_workers)

    def teardown(self, stage: Optional[str] = None):
        pass


