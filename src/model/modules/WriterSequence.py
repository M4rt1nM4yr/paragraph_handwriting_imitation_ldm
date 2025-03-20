import torch
import pytorch_lightning as pl
import torchmetrics
from torch import nn
from src.data.utils.constants import *


from src.model.modules.autoencoder_modules import ResnetBlock, Downsample

import Parameters as pa

class WriterSequence(pl.LightningModule):

    def __init__(self, num_writers=pa.num_writers, in_channels=1, channels = pa.seq_writer_ch,
                 ch_mult=pa.seq_writer_ch_mult,
                 hidden_size=pa.hidden_size, lr = pa.writer_lr, group_norm = 16, only_embed = True,dropout=0.1):
        super(WriterSequence, self).__init__()
        self.lr = lr
        self.only_embed = only_embed
        self.ch = channels
        num_res_blocks = len(ch_mult)-1

        modules = []
        modules.append(torch.nn.Conv2d(in_channels,
                                       ch_mult[0]*channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1))

        for i in range(num_res_blocks):
            modules.append(ResnetBlock(in_channels=ch_mult[i] * channels,
                                       out_channels=ch_mult[i+1] * channels,
                                       temb_channels=0,
                                       dropout=dropout,
                                       group_norm=group_norm))
            if i+1 == num_res_blocks:
                modules.append(Downsample(ch_mult[i + 1] * channels, True, down_sample_factor=2))
            else:
                modules.append(Downsample(ch_mult[i+1] * channels, True, down_sample_factor=4))

        modules.append(ResnetBlock(in_channels=ch_mult[num_res_blocks] * channels,
                                   out_channels=hidden_size,
                                   temb_channels=0,
                                   dropout=dropout,
                                   group_norm=group_norm))

        self.model = nn.Sequential(*modules)

        #torch mean
        self.linear = torch.nn.Linear(hidden_size, hidden_size)  # int(num_writers*hidden_dim))
        self.relu = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(hidden_size, num_writers)

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.acc = torchmetrics.Accuracy(num_classes=num_writers, task='multiclass')

    def forward(self, x):
        seq = self.model(x)

        if self.only_embed:
            return seq

        out = torch.mean(seq, dim=[-1, -2])
        out = self.linear2(self.relu(self.linear(out)))

        return out

    def training_step(self, batch, batch_idx):
        x = batch[IMAGE]
        output = self(x)
        loss = self.criterion(output, batch[WRITER])
        accuracy = self.acc(output, batch[WRITER])
        self.log('train/acc',accuracy)
        self.log('train/loss',loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x = batch[IMAGE]
        output = self(x)
        loss = self.criterion(output, batch[WRITER])
        accuracy = self.acc(output, batch[WRITER])
        self.log('val/acc', accuracy)
        self.log('val/loss', loss)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x = batch[IMAGE]
        output = self(x)
        loss = self.criterion(output, batch[WRITER])
        accuracy = self.acc(output, batch[WRITER])
        self.log("test/loss", loss)
        self.log("test/acc",accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

if __name__ == "__main__":
    wr = WriterSequence(num_writers=10,in_channels=1,hidden_size=32,only_embed=True).cuda()
    x = torch.randn(4,1,512,640).cuda()
    print(wr.forward(x).shape)