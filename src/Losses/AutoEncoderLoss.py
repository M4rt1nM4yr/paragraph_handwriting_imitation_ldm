########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################

import torch
import torch.nn as nn

from thirdparty.VQVAEGAN.TamingTransformers.contperceptual import *
import Parameters as pa
#   taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from src.data.augmentation.Noisy_teacher_forcing import NoisyTeacherForcing
from src.Losses.SmootheCE import SmoothCE
from src.data.utils.constants import *
from src.model.modules.HTR_Writer import HTR_Writer
from src.model.modules.WriterSequence import WriterSequence
import torchmetrics
from src.data.utils.alphabet import Alphabet
from src.utils.utils import *


#This is actually the autoencoder loss

class AutoLoss(nn.Module):
    def __init__(self, disc_start=pa.disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 htr_weight=pa.htr_weight, writer_weight=pa.writer_weight,
                 use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", alphabet_size = pa.alphabet_size, num_writers = pa.num_writers,
                 noisy_teach_prob=pa.noisy_teacher_vqvae,htr_config=None,writer_config=None):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.alphabet = Alphabet()
        self.noisy_teacher = NoisyTeacherForcing(alphabet_size, noisy_teach_prob)

        #Initialising things I need for the writer Loss
        self.writer_weight = writer_weight

        if htr_config is not None:
            self.htrw = instantiate_from_config(htr_config)

            self.htrw.eval()
            self.htrw.freeze()
        else:
                self.htrw = None


        if writer_config is not None:
            self.writer = instantiate_from_config(writer_config)
            self.writer.eval()
            self.writer.freeze()

        else:
                self.writer = None


        self.writerCriterion = torch.nn.CrossEntropyLoss()
        self.writer_weight = writer_weight
        # Initialising things I need for the recognizer Loss
        self.htr_criterion = SmoothCE(mode=0)  # TODO see if this solves problems pa.smooth_ce_mode)
        self.htr_weight = htr_weight

        self.acc = torchmetrics.Accuracy(num_classes=num_writers, task='multiclass')
        self.cer = torchmetrics.CharErrorRate()

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.alphabet = Alphabet()


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight



    def forward(self, inputs, reconstructions,posteriors, sample, optimizer_idx,
                global_step, writers, txt_logits, txt_length, last_layer=None, cond=None, split="train",
                tgt_pad_mask=None, s_mask=None, src_key_padding=None,
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss

        writer_loss = torch.tensor(0.0, device='cuda')
        htr_loss = torch.tensor(0.0, device='cuda')
        htr_rec_loss = torch.tensor(0.0, device='cuda')
        cer = torch.tensor(0.0, device='cuda')
        acc = torch.tensor(0.0, device='cuda')

        if self.htrw is not None:

            # HTR Loss of the reconstruction
            text = self.noisy_teacher(txt_logits, txt_length)
            output_htr, att = self.htrw(tgt_logits=text[:, :-1],memory=reconstructions,
                                             tgt_key_padding_mask=tgt_pad_mask[:, :-1],s_mask=s_mask,
                                             memory_key_padding_mask=src_key_padding)

            correct_logits = torch.flatten(txt_logits[:, 1:])
            output_for_loss = torch.flatten(output_htr, start_dim=0, end_dim=1)

            htr_loss = self.htr_criterion(output_for_loss, correct_logits)

            predicted_logits = torch.argmax(output_htr, dim=2)

            EOS = torch.tensor(3)

            predicted_characters = self.alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
            correct_characters = self.alphabet.batch_logits_to_string_list(txt_logits[:, 1:], [EOS])
            cer = self.cer(predicted_characters[0], correct_characters[0])

            # Add everything to the Reconstruction loss
            nll_loss = nll_loss + self.htr_weight * htr_loss

        if self.writer is not None:
            prediction_writer = self.writer(reconstructions)

            # Writer Loss of the reconstruction
            writer_loss = self.writerCriterion(prediction_writer, writers)
            acc = self.acc(prediction_writer, writers)

            # Add everything to the Reconstruction loss
            nll_loss = nll_loss + self.writer_weight * writer_loss

        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar

        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/writer_loss".format(split): writer_loss.detach().mean(),
                   "{}/HTR_loss".format(split): htr_loss.detach().mean(),
                   "{}/HTR_rec_loss".format(split): htr_rec_loss.detach().mean(),
                   "{}/cer".format(split): cer.detach().mean(),
                   "{}/acc".format(split): acc.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

