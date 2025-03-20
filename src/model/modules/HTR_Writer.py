import math
import copy
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
import torchmetrics

import Parameters
from src.data.utils.constants import *
from src.model.modules.htr import HTR

from torchvision.models import resnet50, ResNet50_Weights
from thirdparty.VQVAEGAN.EncoderDecoder import Encoder
from thirdparty.VQVAEGAN.DiagonalGaussianDistribution import DiagonalGaussianDistribution
from src.data.utils.alphabet import Alphabet
import Parameters as pa
from src.data.augmentation.Noisy_teacher_forcing import NoisyTeacherForcing
from src.Losses.SmootheCE import SmoothCE

from PIL import Image
import torchvision.transforms as T

class HTR_Writer(pl.LightningModule):
    def __init__(self,
                 alphabet_size=pa.alphabet_size,
                 d_model=pa.d_model,
                 nhead=pa.n_head,
                 num_enc_layers = pa.num_enc_layers,
                 num_dec_layers = pa.num_dec_layers,
                 tgt_pe=True,
                 mem_pe=True,
                 dropout=pa.htr_dropout,
                 lr=pa.htr_lr,
                 resolution=pa.resolution,
                 z_channels=pa.d_model,
                 z_double=pa.double_z,
                 ch=pa.ch,
                 ch_mult=pa.htrw_ch_mult,
                 attention_resolution=pa.attention_resolution,
                 noisy_teacher_prob=pa.noisy_teacher,
                 smooth_ce_mode = 0,
                 num_writers = pa.num_writers,
                 encoder_dropout= 0.1,
                 test_with_noisy_teacher=False,
                 patience = 50,
                 writer_analysis_dict=None,
                 long_line_memory=None,
                 remove_new_line_in_model= False,
                 ):

        super(HTR_Writer, self).__init__()
        self.lr = lr

        self.alphabet = Alphabet()

        self.noisy_teacher = NoisyTeacherForcing(A_size=alphabet_size,noise_prob=noisy_teacher_prob)
        self.num_writers = num_writers
        self.test_with_noisy_teacher = test_with_noisy_teacher
        self.patience = patience
        self.remove_new_line_in_model = remove_new_line_in_model

        self.feature_extractor = Encoder(ch=ch, out_ch=1,dropout=encoder_dropout, num_res_blocks=2, attn_resolutions=attention_resolution,
                                             in_channels=1, resolution=resolution, z_channels=z_channels,
                                            ch_mult=ch_mult, double_z=z_double)


        self.quant_conv = torch.nn.Conv2d(z_channels, d_model, 1)

        # for classification of the letters

        self.htr = HTR(alphabet_size=alphabet_size, nhead=nhead, d_model=d_model, num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, tgt_pe=tgt_pe,
                       mem_pe=mem_pe, dropout=dropout, lr=lr)

        self.criterionHTR = SmoothCE(mode=smooth_ce_mode)
        self.cer = torchmetrics.CharErrorRate()

        #extra anaylsis stuff. Just ignore it
        self.test_cers = []
        self.test_texts = []

        self.writer_analysis_dict = writer_analysis_dict
        self.clean_cers = []
        self.id_cers = []

        self.long_line_memory = long_line_memory
        self.long_line_cers = []
        self.short_line_cers = []

    #Feature Extractor
    def encode(self, memory):

        memory = self.feature_extractor(memory)
        memory = self.quant_conv(memory)

        return memory


    def forward(self, tgt_logits, memory, s_mask= None,tgt_key_padding_mask= None,memory_key_padding_mask=None, **kwargs):

        memory = self.encode(memory)
        output_htr, attention_htr, att_img = self.htr(tgt_logits,memory=memory,tgt_mask=s_mask,
                                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                                          memory_key_padding_mask=memory_key_padding_mask)

        return output_htr, attention_htr
    def training_step(self, batch, batch_idx):

        x = batch[IMAGE]
        text = self.noisy_teacher(batch[TEXT_LOGITS_S2S],batch[UNPADDED_TEXT_LEN])

        output_htr, att = self(tgt_logits=text[:,:-1], memory= x, s_mask=batch[TGT_MASK],
                                tgt_key_padding_mask=batch[TGT_KEY_PADDING_MASK][:,:-1],memory_key_padding_mask =batch[SRC_KEY_PADDING])

        #Caclulating the HTR Loss
        output_for_loss = torch.flatten(output_htr, start_dim=0, end_dim=1)
        correct_logits = torch.flatten(batch[TEXT_LOGITS_S2S][:, 1:])
        loss = self.criterionHTR(output_for_loss, correct_logits)

        predicted_logits = torch.argmax(output_htr, dim=2)

        EOS = torch.tensor(3)

        predicted_characters = self.alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
        correct_characters = self.alphabet.batch_logits_to_string_list(batch[TEXT_LOGITS_S2S][:, 1:], [EOS])
        cer = self.cer(predicted_characters[0], correct_characters[0])
        self.log('train/cer', cer)
        self.log('train/loss', loss)

        return {'loss': loss,'cer': cer.detach()}


    def validation_step(self, batch, batch_idx):
        x = batch[IMAGE]
        text = self.noisy_teacher(batch[TEXT_LOGITS_S2S], batch[UNPADDED_TEXT_LEN])

        output_htr, att= self(tgt_logits=text[:, :-1], memory=x,s_mask=batch[TGT_MASK],
                              tgt_key_padding_mask=batch[TGT_KEY_PADDING_MASK][:,:-1],
                              memory_key_padding_mask=batch[SRC_KEY_PADDING])

        # Caclulating the HTR Loss
        output_for_loss = torch.flatten(output_htr, start_dim=0, end_dim=1)
        correct_logits = torch.flatten(batch[TEXT_LOGITS_S2S][:, 1:])
        loss = self.criterionHTR(output_for_loss, correct_logits)

        predicted_logits = torch.argmax(output_htr, dim=2)

        EOS = torch.tensor(3)

        predicted_characters = self.alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
        correct_characters = self.alphabet.batch_logits_to_string_list(batch[TEXT_LOGITS_S2S][:, 1:], [EOS])
        cer = self.cer(predicted_characters[0], correct_characters[0])
        self.log('val/cer',cer)
        self.log('val/loss',loss)

        return {'v_loss': loss,'v_cer': cer.detach()}


    def test_step(self,batch, batch_idx):


        text_shape = batch[TEXT_LOGITS_S2S].shape[1]
        x = batch[IMAGE]
        EOS = torch.tensor(3)


        if self.test_with_noisy_teacher:
            input_text = self.noisy_teacher(batch[TEXT_LOGITS_S2S], batch[UNPADDED_TEXT_LEN])
            pred, att_img = self(tgt_logits=input_text[:, :-1], memory=x,s_mask=batch[TGT_MASK],
                                                           tgt_key_padding_mask=batch[TGT_KEY_PADDING_MASK][:,:-1],
                                                           memory_key_padding_mask=batch[SRC_KEY_PADDING])
            predicted_logits = torch.argmax(pred, dim=2)

        else:
            x = self.encode(x)
            pred, predicted_logits = self.inference_htr(x, max_char_len=text_shape,memory_key_padding_mask=batch[SRC_KEY_PADDING])#,debug_correct_logits=input_text )

        correct_logits = torch.flatten(batch[TEXT_LOGITS_S2S][:, 1:])
        pred_flattened = torch.flatten(pred, start_dim=0, end_dim=1)
        loss = self.criterionHTR(pred_flattened, correct_logits)

        predicted_characters = self.alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
        correct_characters = self.alphabet.batch_logits_to_string_list(batch[TEXT_LOGITS_S2S][:, 1:], [EOS])

        correct_characters_list = []
        predicted_characters_list = []

        predicted_characters = predicted_characters[0]
        correct_characters = correct_characters[0]

        if self.remove_new_line_in_model:
            for i in range(len(predicted_characters)):
                correct_characters_list.append(correct_characters[i].replace('\n', ''))
                predicted_characters_list.append(predicted_characters[i].replace('\n', ''))

            correct_characters = correct_characters_list
            predicted_characters = predicted_characters_list

        cer = self.cer(predicted_characters, correct_characters)

        self.log("{}/loss".format("test"),loss)
        self.log("{}/cer".format("test"),cer)
        self.test_cers.append(cer.cpu().numpy().item())
        self.test_texts.append(batch[TEXT][0])

        if self.writer_analysis_dict is not None and batch[WRITER].shape[0]==1:
            writer = batch[WRITER][0].cpu().numpy().item()
            if self.writer_analysis_dict[writer] > 1:
                self.clean_cers.append(cer.cpu().numpy().item())
            else:
                self.id_cers.append(cer.cpu().numpy().item())
        if self.long_line_memory is not None and len(batch["name"])==1:
            if self.long_line_memory[batch["name"][0]]:
                self.long_line_cers.append(cer.cpu().numpy().item())
            else:
                self.short_line_cers.append(cer.cpu().numpy().item())

        return {'t_loss': loss,'t_cer': cer.detach()}

    @torch.no_grad()
    def inference_htr(
            self,
            x,
            img_width=None,
            max_char_len=777,
            memory_key_padding_mask=None):

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.reshape(x.shape[0], -1)

        # ENCODER
        enc_out = x
        enc_out,_ = self.htr.transformer_encode(enc_out,memory_key_padding_mask=memory_key_padding_mask)
        # DECODER
        pred_logits = (torch.ones(size=(x.shape[0], max_char_len)) * self.alphabet.toPosition[PAD]).long()
        pred = torch.ones(size=(x.shape[0], max_char_len, len(self.alphabet.toPosition)))
        pred_logits[:, 0] = (torch.ones(size=pred_logits[:, 0].shape) * self.alphabet.toPosition[START_OF_SEQUENCE]).long()
        if x.is_cuda:
            pred_logits = pred_logits.cuda()
            pred = pred.cuda()


        for i in range(1, max_char_len):
            dec_out, att_image = self.htr.transformer_decode(
                memory=enc_out,
                tgt_logits=pred_logits[:, :i],
                memory_key_padding_mask=memory_key_padding_mask
            )
            pred[:, i] = dec_out[:, -1, :]
            pred_logits[:, i] = torch.argmax(dec_out[:, -1, :], dim=-1)

        return pred[:,1:],pred_logits[:,1:]

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=self.patience)
        metric = "val/cer"
        interval = "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": metric,
                "interval": interval,
                "frequency": 1,
            }
        }


