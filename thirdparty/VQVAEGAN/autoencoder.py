########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import math
from contextlib import contextmanager

from torch.optim.lr_scheduler import LambdaLR

from thirdparty.VQVAEGAN.TamingTransformers.VectorQuantizer import VectorQuantizer2 as VectorQuantizer

from thirdparty.VQVAEGAN.EncoderDecoder import Encoder, Decoder
from thirdparty.VQVAEGAN.DiagonalGaussianDistribution import DiagonalGaussianDistribution
from src.Losses.LossVQVAEGAN import VQVAELoss
from src.Losses.AutoEncoderLoss import AutoLoss
# from ldm.util import instantiate_from_config
from thirdparty.VQVAEGAN.lr_scheduler import LambdaLinearScheduler
from src.data.utils.constants import *
from thirdparty.VQVAEGAN.ema import *

import Parameters as pa
from PIL import Image
import torchvision.transforms as T

#TODO this should be deleted but can't due to dependencies...
class VQModel(pl.LightningModule):
    def __init__(self,
                 n_embed=pa.n_embedded,
                 embed_dim=pa.embedded_dim,
                 ch=pa.ch,
                 in_ch_enc=1,
                 out_ch_dec=1,
                 enc_res_blocks=2,
                 att_resolutions=[],
                 resolution=pa.resolution,
                 z_channels=pa.z_channels,
                 dropout=pa.vq_vae_dropout,
                 ckpt_path=pa.checkpoint_VQVAE,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 alphabet_size=pa.alphabet_size,
                 num_writers=pa.num_writers,
                 in_channels=pa.in_channels,
                 hidden_size=pa.hidden_size,
                 ch_mult=pa.ch_mult,
                 learning_rate=0.0001,
                 writer_weight=pa.writer_weight,
                 htr_weight=pa.htr_weight,
                 disc_start=pa.disc_start,
                 disc_weight=pa.disc_weight,
                 double_z=pa.double_z,
                 checkpoint_htrw = pa.checkpoint_htrw,
                 ch_mult_htr = pa.ch_mult,# (1,2,4,8,8),
                 quantizer_legacy = pa.quantizer_legacy,
                 d_model = pa.d_model
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.learning_rate = learning_rate

        self.encoder = Encoder(ch=ch, out_ch=out_ch_dec, num_res_blocks=enc_res_blocks,
                               attn_resolutions=att_resolutions, dropout=dropout, z_channels=z_channels,
                               resolution=resolution, in_channels=in_ch_enc, ch_mult=ch_mult, double_z=double_z)

        self.decoder = Decoder(ch=ch, out_ch=out_ch_dec, num_res_blocks=enc_res_blocks,
                               attn_resolutions=att_resolutions, dropout=dropout, z_channels=z_channels,
                               resolution=resolution, in_channels=in_ch_enc, ch_mult=ch_mult, double_z=double_z)

        self.loss = VQVAELoss( checkpoint_htrw=checkpoint_htrw,
                                 alphabet_size=alphabet_size, num_writers=num_writers, ch = ch,
                                 in_channels=embed_dim, hidden_size=hidden_size,
                                 ch_mult=ch_mult_htr, writer_weight=writer_weight, htr_weight=htr_weight,
                                 disc_weight=disc_weight, z_channels=z_channels,
                                 disc_start=disc_start,n_classes=n_embed,z_double = double_z,embedded_dim=embed_dim,
                                 d_model=d_model)  # instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape,legacy=quantizer_legacy)
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind), z = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind, z
        return dec, diff, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # Data is already in the correct format
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = batch[IMAGE]#self.get_input(batch, self.image_key)
        writers = batch[WRITER]
        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        src_padding = batch[SRC_KEY_PADDING]

        xrec, qloss, ind, z = self(x, return_pred_indices=True)
        #TODO maybe I can delete this line and z?
        with torch.no_grad():
            sample2 = None#self.encode_to_prequant(xrec)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, z, sample2, optimizer_idx, self.global_step,
                                            writers=writers, txt_logits=txt_logits,txt_length = txt_length,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind,tgt_pad_mask=batch[TGT_KEY_PADDING_MASK],s_mask =batch[TGT_MASK],src_key_padding=src_padding)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, z, sample2, optimizer_idx, self.global_step,
                                                writers=writers, txt_logits=txt_logits, txt_length = txt_length,
                                                last_layer=self.get_last_layer(), split="train",tgt_pad_mask=batch[TGT_KEY_PADDING_MASK],s_mask =batch[TGT_MASK],src_key_padding=src_padding)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind, z = self(x, return_pred_indices=True)
        writers = batch[WRITER]
        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        src_padding = batch[SRC_KEY_PADDING]

        with torch.no_grad():
            sample2 = None#self.encode_to_prequant(xrec)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, z, sample2, 0,
                                        self.global_step,
                                        writers=writers, txt_logits=txt_logits,txt_length = txt_length,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=ind,tgt_pad_mask=batch[TGT_KEY_PADDING_MASK],s_mask =batch[TGT_MASK],src_key_padding=src_padding)


        discloss, log_dict_disc = self.loss(qloss, x, xrec, z, sample2, 1,
                                            self.global_step,
                                            writers=writers, txt_logits=txt_logits,txt_length = txt_length,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=ind,tgt_pad_mask=batch[TGT_KEY_PADDING_MASK],s_mask =batch[TGT_MASK],src_key_padding=src_padding)
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # TODO might have to change that ?
        #if version.parse(pl.__version__) >= version.parse('1.4.0'):
        del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    #Evaluates only on Rec loss
    def test_step(self, batch, batch_idx, suffix="") :
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind, z = self(x, return_pred_indices=True)

        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        tgt_pad_mask = batch[TGT_KEY_PADDING_MASK]
        s_mask = batch[TGT_MASK]

        cer_prime = torch.tensor(0.0, device='cuda')
        acc_prime = torch.tensor(0.0, device='cuda')
        mse_loss = torch.tensor(0.0, device='cuda')

        if self.prime_latent:
            # TODO prime Latent Space here
            sample = z#self.loss.to_prime(z)
            text_shape = batch[TEXT_LOGITS_S2S].shape[1]

            output_htr_prime , predicted_logits_prime = self.loss.htrw_prime.inference_htr(sample, max_char_len=text_shape)

            output_writer_prime, embedding_prime = self.loss.htrw_prime.inference_writer(sample)
            encoding_x = self.loss.htrw_loss.encode(x)
            output_writer, embedding = self.loss.htrw_loss.inference_writer(encoding_x)
            mse_loss = torch.nn.functional.mse_loss(embedding_prime, embedding)

            EOS = torch.tensor(3)
            correct_characters = self.loss.alphabet.batch_logits_to_string_list(txt_logits[:, 1:], [EOS])

            predicted_logits_prime = torch.argmax(output_htr_prime, dim=2)
            predicted_characters_prime = self.loss.alphabet.batch_logits_to_string_list(predicted_logits_prime, [EOS])
            cer_prime = self.loss.cer_prime(predicted_characters_prime[0], correct_characters[0])
            self.log("test/cer_prime", cer_prime)
            self.log("test/mse_loss_prime",mse_loss)


        rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
        rec_mean_loss = torch.mean(rec_loss)
        self.log("test/rec_loss", rec_mean_loss)
        return {'rec_loss': rec_mean_loss, 'cer_prime': cer_prime, 'mse_loss_prime': mse_loss}

    def test_epoch_end(self, outputs) :
        average_loss = torch.stack([l['rec_loss'] for l in outputs]).mean()
        self.log("test/rec_loss", average_loss.detach())

        average_cer = torch.stack([l['cer_prime'] for l in outputs]).mean()
        self.log("test/cer_prime", average_cer.detach())

        average_acc = torch.stack([l['mse_loss_prime'] for l in outputs]).mean()
        self.log("test/mse_loss_prime", average_acc.detach())

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        if self.prime_latent:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quantize.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters())+
                                      list(self.loss.htrw_prime.parameters()),
                                      lr=lr_g, betas=(0.5, 0.9))
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))



        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = LambdaLinearScheduler(warm_up_steps=[10000], cycle_lengths=[10000000000000], f_start=[1.e-6],
                                              f_max=[1.], f_min=[1.])  # instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim=pa.embedded_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class VQModelInterfacePost(VQModel):
    def __init__(self, embed_dim=pa.embedded_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant,_, _ = self.quantize(h)
        return quant

    def decode(self, h, force_not_quantize=False):
        quant = self.post_quant_conv(h)
        dec = self.decoder(quant)
        return dec

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim=pa.embedded_dim,
                 ckpt_path=pa.ckpt_autoKL,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ch=pa.ch,
                 in_ch_enc=1,
                 out_ch_dec=1,
                 enc_res_blocks=2,
                 att_resolutions=[],
                 resolution=pa.resolution,
                 z_channels=pa.z_channels,
                 dropout=pa.vq_vae_dropout,
                 alphabet_size=pa.alphabet_size,
                 num_writers=pa.num_writers,
                 ch_mult=pa.ch_mult,
                 learning_rate=0.0001,
                 writer_weight=pa.writer_weight,
                 htr_weight=pa.htr_weight,
                 disc_start=pa.disc_start,
                 disc_weight=pa.disc_weight,
                 kl_weight=0.000001,
                 htr_config=None,
                 writer_config=None,
                 **kwargs):
        super().__init__()

      # self.automatic_optimization = False

        self.image_key = image_key
        self.learning_rate = learning_rate

        self.encoder = Encoder(ch=ch, out_ch=out_ch_dec, num_res_blocks=enc_res_blocks,
                               attn_resolutions=att_resolutions, dropout=dropout, z_channels=z_channels,
                               resolution=resolution, in_channels=in_ch_enc, ch_mult=ch_mult,double_z=True)
        self.decoder = Decoder(ch=ch, out_ch=out_ch_dec, num_res_blocks=enc_res_blocks,
                               attn_resolutions=att_resolutions, dropout=dropout, z_channels=z_channels,
                               resolution=resolution, in_channels=in_ch_enc, ch_mult=ch_mult,double_z= True)
        self.loss = AutoLoss(alphabet_size=alphabet_size, num_writers=num_writers,
                             writer_weight=writer_weight, htr_weight=htr_weight,
                             disc_weight=disc_weight, kl_weight=kl_weight, disc_start=disc_start,
                             htr_config=htr_config,writer_config=writer_config,**kwargs)

        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # data is already in the correct format
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
      # optimizer_idx = 1

        inputs = batch[IMAGE]  # self.get_input(batch, self.image_key)
        reconstructions, posterior, z = self(inputs)

        writers = batch[WRITER]
        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        src_padding = batch[SRC_KEY_PADDING]
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, z, 0,
                                            self.global_step, writers=writers, txt_logits=txt_logits, txt_length=txt_length,
                                            tgt_pad_mask = batch[TGT_KEY_PADDING_MASK],s_mask=batch[TGT_MASK],src_key_padding=src_padding,
                                            last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss


    def validation_step(self, batch, batch_idx):
        inputs = batch[IMAGE]  # self.get_input(batch, self.image_key)
        writers = batch[WRITER]
        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        src_padding = batch[SRC_KEY_PADDING]

        reconstructions, posterior, z = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, z, 0,
                                        self.global_step, writers=writers, txt_logits=txt_logits, txt_length=txt_length,
                                        tgt_pad_mask = batch[TGT_KEY_PADDING_MASK],s_mask=batch[TGT_MASK],src_key_padding=src_padding,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, z, 1,
                                            self.global_step, writers=writers, txt_logits=txt_logits, txt_length=txt_length,
                                            tgt_pad_mask = batch[TGT_KEY_PADDING_MASK],s_mask=batch[TGT_MASK],src_key_padding=src_padding,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    # Evaluates only on Rec loss
    def test_step(self, batch, batch_idx, suffix=""):
        x = batch[IMAGE]
        txt_logits = batch[TEXT_LOGITS_S2S]
        txt_length = batch[UNPADDED_TEXT_LEN]
        tgt_pad_mask = batch[TGT_KEY_PADDING_MASK]
        s_mask = batch[TGT_MASK]

        mse_loss = torch.tensor(0.0, device='cuda')
        xrec, posterior, z = self(x)


        rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
        rec_mean_loss = torch.mean(rec_loss)
        self.log("test/rec_loss", rec_mean_loss)
        return {'rec_loss': rec_mean_loss}



    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters()),
                                      lr=lr, betas=(0.5, 0.9))
      #  opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
     #                               lr=lr, betas=(0.5, 0.9))
        return opt_ae #[opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
