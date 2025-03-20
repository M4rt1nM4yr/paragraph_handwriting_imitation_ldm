import math
import copy
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from src.data.utils.constants import *


import Parameters as pa
class HTR(pl.LightningModule):
    def __init__(self,
                 alphabet_size=pa.alphabet_size,
                 d_model=pa.d_model,
                 nhead=pa.n_head,
                 num_enc_layers = pa.num_enc_layers,
                 num_dec_layers = pa.num_dec_layers,
                 tgt_pe=True,
                 mem_pe=True,
                 dropout=pa.htr_dropout,
                 lr=pa.htr_lr):
        super(HTR, self).__init__()
        self.text_embedding = nn.Embedding(alphabet_size,d_model)
        self.d_model = d_model
        if tgt_pe != None:
            self.tgt_pe = PositionalEncoding1D(d_model=d_model, dropout=dropout)
        else: self.tgt_pe=None
        if mem_pe != None:
            self.mem_pe = A2DPE(d_model=d_model, dropout=dropout)
        else: self.mem_pe=None
        self.lr = lr


        self.transformEnc = TransformerEncoderStack(TransformerEncoder(d_model=d_model,nhead=nhead),num_layers=num_enc_layers,d_model=d_model)
        normDec = LayerNorm(d_model)
        self.dec = TransformerDecoder(TransformerDecoderLayer(d_model=d_model, nhead=nhead), num_layers=num_dec_layers,norm=normDec)

        self.fc = torch.nn.Linear(d_model,alphabet_size)

    """
    Transformer Encoder step
    Adds PE to the memory, forwards memory into the transformer encoder
     and reshapes it into a sequence for the decoder step
    """

    def transformer_encode(self, memory, memory_key_padding_mask=None):

        if self.mem_pe != None:
            memory = self.mem_pe(memory)

        mem_s = memory.shape
        memory = memory.reshape(mem_s[0], mem_s[1], -1).permute(0, 2, 1)  # -> N, (HxW), F
        memory = memory.permute(1, 0, 2)



        memory, attention_img = self.transformEnc(memory=memory,memory_key_padding_mask=memory_key_padding_mask)

        return memory, attention_img

    """
        Transformer Decoder step
        Adds PE to the logits, forwards everything into the transformer decoder
        returns pred prior to softmax for the single characters
        """
    def transformer_decode(self,memory,tgt_logits,memory_key_padding_mask=None,**kwargs):
        tgt = self.text_embedding(tgt_logits)
        if self.tgt_pe != None:
            tgt = self.tgt_pe(tgt)
        preds, attention_matrix = self.dec(tgt=tgt.permute(1, 0, 2), memory=memory,
                                           memory_key_padding_mask=memory_key_padding_mask, **kwargs)

        output = preds.permute(1, 0, 2)
        output = self.fc(output)

        return output, attention_matrix

    def forward(self, tgt_logits, memory,memory_key_padding_mask = None, **kwargs):
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.reshape(memory.shape[0], -1)

        memory, attention_img = self.transformer_encode(memory, memory_key_padding_mask)
        output,attention_matrix = self.transformer_decode(memory,tgt_logits,memory_key_padding_mask,**kwargs)

        return output, attention_matrix, attention_img


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        attention_maps = list()

        for mod in self.layers:
            output, attention = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attention_maps.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        attention_maps = torch.stack(attention_maps).permute(1,0,2,3)
        return output, attention_maps

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2,attention = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention

# Encoder like
# https://arxiv.org/abs/2005.13044
class TransformerEncoderStack(nn.Module):
    r"""TransformerEncoderStack is a stack of N TransformerEncoders

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, d_model=pa.d_model):
        super(TransformerEncoderStack, self).__init__()
   #     self.normIn = LayerNorm(d_model)

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.normOut = LayerNorm(d_model)

    def forward(self, memory, memory_mask=None, memory_key_padding_mask=None):
        # type: (Tensor,  Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the encoder layer in turn.

        Args:
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = memory
        attention_maps = list()

   #     output = self.normIn(output)


        for mod in self.layers:
            output, attention = mod(output,
                         memory_mask=memory_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attention_maps.append(attention)

        output = self.normOut(output)
        attention_maps = torch.stack(attention_maps).permute(1,0,2,3)

        return output, attention_maps

# TransformerEncoder based on the paper Transformer-Based Approach for Joint Handwriting
# and Named Entity Recognition in Historical documents
class TransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward,d_model)
        self.dropout = Dropout(dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, memory, memory_mask=None, memory_key_padding_mask=None):
        # type: (Tensor,  Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the encoder layer.

        Args:
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        memory2,attention = self.self_attn(memory, memory, memory)
        memory = memory + self.dropout1(memory2)
        memory = self.norm1(memory)

        memory2 = self.linear2(self.dropout(self.activation(self.linear1(memory))))
        memory = memory + self.dropout2(memory2)
        memory = self.norm2(memory)
        return memory, attention




class PositionalEncoding1D(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=1000, positional_scaler =1.0):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, device='cuda')
        position = torch.arange(0, max_len,device='cuda').unsqueeze(1) * positional_scaler
        div_term = torch.exp(torch.arange(0, d_model, 2,device='cuda') *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        shape of x has to be: (N, S, F)   # N = batch size, S = sequence length, F = feature dimension
        """
        pe = Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        x = x + pe
        return self.dropout(x)

class A2DPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, positional_scaler = 1.0):
        super(A2DPE, self).__init__()
        self.alpha_fc = PEScaling()
        self.beta_fc = PEScaling()
        self.P = PositionalEncoding1D(d_model=d_model, dropout=dropout, positional_scaler=positional_scaler).pe.squeeze()

    def forward(self, x):
        if x.is_cuda:
            self.P = self.P.cuda()
        else:
            self.P = self.P.cpu()
        pe = torch.stack([self.P]*x.shape[0])
        alpha, beta = self.alpha_fc(x), self.beta_fc(x)
        Ph = alpha*pe[:,:x.shape[-2]]
        Pw = beta*pe[:,:x.shape[-1]]
        Ph = torch.repeat_interleave(Ph, x.shape[-1], dim=1)
        Pw = Pw.repeat(1, x.shape[-2], 1)
        POS = (Ph+Pw).reshape(x.shape)
        x = POS + x
        return x

class PEScaling(nn.Module):
    def __init__(self):
        super(PEScaling, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        # global avg pooling
        E = x.mean(-1).mean(-1).mean(-1).unsqueeze(-1)
        return self.sigmoid(self.linear2(self.relu(self.linear1(E)))).unsqueeze(-1)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = ~(torch.from_numpy(subsequent_mask) == 0).squeeze(0)
    matrix_ninf = torch.ones(()) * float('-inf')
    matrix_zeros = torch.zeros(()).float()
    subsequent_mask = torch.where(subsequent_mask,matrix_ninf,matrix_zeros)
    return subsequent_mask




