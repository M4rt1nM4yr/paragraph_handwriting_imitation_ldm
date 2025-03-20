import math
import copy
import torch
import torchmetrics
from torch.autograd import Variable
from torch import nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from src.data.utils.constants import *


from src.model.modules.htr import PositionalEncoding1D, A2DPE, TransformerDecoderLayer,TransformerDecoder
from src.model.modules.WriterSequence import WriterSequence
import Parameters as pa

"""
    This is the text-style encoder that facilitates CROSS CONDITIONING
"""

class TextStyleSimple(nn.Module): #CROSS-ATTENTION MODULE
    def __init__(self,
                 channels = pa.cond_channels,
                 d_model = pa.unet_context_dim,
                 alphabet_size = pa.alphabet_size,
                 lr = 0.0001,
                 decoder_nhead = 4,
                 ckpt_seq_w = pa.checkpoint_seq_writer,
                 hidden_size = pa.cond_hidden,
                 num_writers = 1045,
                 cond_stage_nr_encoders = pa.cond_stage_nr_encoders,
                 dropout=pa.cond_stage_dropout,
                 cond_ch_mult = pa.cond_ch_mult):
        super(TextStyleSimple, self).__init__()
        self.lr = lr
        self.channels = channels
        self.nr_encoders = cond_stage_nr_encoders

        #Text Processing
        self.text_embedding = nn.Embedding(alphabet_size, channels)
        self.pe1d = PositionalEncoding1D(d_model=channels)

        #Writer Processing
        self.writer = WriterSequence.load_from_checkpoint(ckpt_seq_w, num_writers=num_writers,ch_mult=cond_ch_mult,
                                                          hidden_size=hidden_size)
        self.writer.cuda()

        self.hidden_to_c = torch.nn.Conv2d(in_channels=hidden_size,
                                           out_channels=channels,
                                           stride=1,
                                           kernel_size=3,
                                           padding=1)
        self.pe2d = A2DPE(d_model=channels)


        self.transformer_decoder = TransformerDecoder(TransformerDecoderLayer(d_model=channels,nhead=decoder_nhead,
                                                                                  dropout=dropout),
                                                               num_layers=self.nr_encoders)

        # To Context Dim
        self.to_context_dim = torch.nn.Linear(channels,d_model)



    #TODO seq writer has probably too much room
    def forward(self, text, style_sample, tgt_key_padding_mask=None,style_padding_mask=None, **kwargs):

        #Embeddings are (B,SeqLen, channels)
        tgt = self.text_embedding(text)
        tgt = self.pe1d(tgt)

        style_embed = self.writer(style_sample)
        style_embed = self.hidden_to_c(style_embed)
        style_embed = self.pe2d(style_embed)
        style_embed = torch.flatten(style_embed,start_dim=2)
        style_embed = style_embed.permute(2,0,1)

        #TODO: important ! also concat padding with more zeros
        orginal_tgt = tgt

        if style_padding_mask is not None:
            style_padding_mask = torch.flatten(style_padding_mask,start_dim=1)
        tgt = tgt.permute(1, 0, 2)
        tgt = self.transformer_decoder(tgt=tgt,memory=style_embed, tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=style_padding_mask)[0]
        tgt = tgt.permute(1, 0, 2)
        tgt_with_style = tgt + orginal_tgt

        tgt_with_style = self.to_context_dim(tgt_with_style)

        return tgt_with_style, tgt_key_padding_mask
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer


