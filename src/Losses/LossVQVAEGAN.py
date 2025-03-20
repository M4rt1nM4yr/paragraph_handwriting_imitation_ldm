########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################
#Ignore this file, it's only here for legacy reasons....

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from thirdparty.VQVAEGAN.TamingTransformers.contperceptual import *

import Parameters as pa
from src.model.modules.htr import subsequent_mask
from src.data.utils.alphabet import Alphabet
from src.data.augmentation.Noisy_teacher_forcing import NoisyTeacherForcing
from src.Losses.SmootheCE import SmoothCE
from src.data.utils.constants import *
#   taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
#from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from src.model.modules.HTR_Writer import HTR_Writer
import torchmetrics
from src.data.utils.alphabet import Alphabet


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class VQVAELoss(nn.Module):
    def __init__(self, disc_start=pa.disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 htr_weight=pa.htr_weight, writer_weight=pa.writer_weight,
                 use_actnorm=False, disc_conditional=False, ch = pa.ch,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, z_channels = pa.z_channels,
                 pixel_loss="l1", alphabet_size = pa.alphabet_size, num_writers = pa.num_writers,in_channels = pa.in_channels,hidden_size=pa.hidden_size, ch_mult =pa.ch_mult,
                 z_double = pa.double_z , noisy_teach_prob = pa.noisy_teacher_vqvae,
                 checkpoint_htrw = pa.checkpoint_htrw,embedded_dim = pa.embedded_dim,d_model=pa.d_model):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]

        # Initialising things I need for the writer Loss
        self.alphabet = Alphabet()
        self.noisy_teacher = NoisyTeacherForcing(alphabet_size,noisy_teach_prob)
        self.prime_latent = prime_latent
        self.start_htr_weight = start_htr_weight

        if checkpoint_htrw is not None:
            self.htrw_loss = HTR_Writer.load_from_checkpoint(checkpoint_htrw,num_writers=num_writers,
                                                             d_model = d_model,
                                                             in_channels=in_channels, FeatureExtractor=True,
                                                             hidden_size=hidden_size, ch_mult=ch_mult,
                                                             auto_encoder_used=False, z_double=z_double,ch=ch,
                                                            z_channels=d_model)
            self.htrw_loss.eval()
            self.htrw_loss.freeze()
        else:
            self.htrw_loss = None

        if prime_latent and checkpoint_htrw is not None:

            self.htrw_prime = HTR_Writer.load_from_checkpoint(checkpoint_htrw,num_writers=num_writers,
                                                             d_model =d_model,
                                                             in_channels=in_channels, FeatureExtractor=False,
                                                             hidden_size=hidden_size, ch_mult=ch_mult,
                                                               auto_encoder_used=False, z_double=z_double,ch=ch,
                                                               z_channels=z_channels,strict=False,vertical_start=start_vertical_sampling)
            self.htr_criterion_prime = SmoothCE(mode=1)#TODO see if this fixes problems? pa.smooth_ce_mode)
            self.cer_prime = torchmetrics.CharErrorRate()
            self.prime_style = prime_style


            #TOOD this can be literally removed. Will checkpoints still work?
            #if prime_style :
            #    self.prime_style_criterion = torch.nn.CrossEntropyLoss()



        else:
            self.htrw_prime = None



        # cycle consistency loss of the writer embeddings
        self.cycle_weight = cycle_weights
        self.prime_weight = prime_weight

        self.writerCriterion = torch.nn.CrossEntropyLoss()
        self.writer_weight = writer_weight
        # Initialising things I need for the recognizer Loss
        self.htr_criterion = SmoothCE(mode=0)#TODO see if this solves problems pa.smooth_ce_mode)
        self.htr_weight = htr_weight

        self.acc = torchmetrics.Accuracy(num_classes=num_writers, task='multiclass')
        self.cer = torchmetrics.CharErrorRate()

        self.codebook_weight = codebook_weight

        #pixel weights
        self.pixel_weight = pixelloss_weight
        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes
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


    #TODO this is experimental. If it works well make it with parameters
    def weight_decay_for_htr(self,current_step):
        return self.htr_weight



    def forward(self, codebook_loss, inputs, reconstructions,sample,  sample2, optimizer_idx,
                global_step, writers, txt_logits,txt_length, last_layer=None, cond=None, split="train", predicted_indices=None,
                tgt_pad_mask = None, s_mask = None, src_key_padding = None):
        if not (codebook_loss is not None):
            codebook_loss = torch.tensor([0.],device='cuda')
        #rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)


        recWHTR_loss = nll_loss
        #nll_loss = recWHTR_loss
        #nll_loss = torch.mean(nll_loss)

        cycleConLoss = torch.tensor(0.0,device='cuda')
        primeLoss = torch.tensor(0.0,device='cuda')

        writer_loss = torch.tensor(0.0,device='cuda')
        htr_loss = torch.tensor(0.0,device='cuda')
        htr_rec_loss = torch.tensor(0.0,device='cuda')
        cer = torch.tensor(0.0,device='cuda')
        acc = torch.tensor(0.0,device='cuda')
        cer_prime = torch.tensor(0.0,device='cuda')

        if(self.htrw_loss is not None):

            #HTR Loss of the reconstruction
            text = self.noisy_teacher(txt_logits,txt_length)
            output_htr, att, output_writer, embedding_writer = self.htrw_loss(tgt_logits=text[:, :-1], memory=reconstructions, tgt_key_padding_mask=tgt_pad_mask[:,:-1],s_mask=s_mask,src_key_padding=src_key_padding)
            correct_logits = torch.flatten(txt_logits[:, 1:])
            output_for_loss = torch.flatten(output_htr, start_dim=0, end_dim=1)

            htr_loss = self.htr_criterion(output_for_loss, correct_logits)

            predicted_logits = torch.argmax(output_htr, dim=2)

            EOS = torch.tensor(3)

            predicted_characters = self.alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
            correct_characters = self.alphabet.batch_logits_to_string_list(txt_logits[:, 1:], [EOS])
            cer = self.cer(predicted_characters[0], correct_characters[0])



            # Writer Loss of the reconstruction
            writer_loss = self.writerCriterion(output_writer, writers)
            acc = self.acc(output_writer, writers)

            # prime Latent space. Right now this is gonna be False
            if self.prime_latent:
                # TODO prime Latent Space here

                output_htr_prime, _, _, embedding_prime_writer =self.htrw_prime(tgt_logits=text[:, :-1], memory=sample, tgt_key_padding_mask=tgt_pad_mask[:,:-1],s_mask=s_mask)

                #calculating losses
                #HTR
                output_for_loss_prime = torch.flatten(output_htr_prime, start_dim=0, end_dim=1)
                htr_loss_prime = self.htr_criterion_prime(output_for_loss_prime, correct_logits)
                #Embedding of Writer is the same for HTR and Encoder
               # cycleConLoss = torch.nn.functional.mse_loss(embedding_writer, embedding_prime_writer)

                if self.prime_style :
                    cycleConLoss = torch.nn.functional.mse_loss(embedding_writer, embedding_prime_writer)

                primeLoss = htr_loss_prime * self.prime_weight + cycleConLoss * self.cycle_weight

                predicted_logits_prime = torch.argmax(output_htr_prime, dim=2)
                predicted_characters_prime = self.alphabet.batch_logits_to_string_list(predicted_logits_prime, [EOS])
                cer_prime = self.cer_prime(predicted_characters_prime[0], correct_characters[0])


                # TODO does this have a point?
                # with torch.no_grad():
                #     _, writer_embedding2 = self.writer_loss(sample2)

        recWHTR_loss = recWHTR_loss + self.weight_decay_for_htr(global_step) *htr_loss + self.writer_weight * writer_loss + primeLoss

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

            try:
                d_weight = self.calculate_adaptive_weight(recWHTR_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = recWHTR_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/writer_loss".format(split): writer_loss.detach().mean(),
                   "{}/cycle_loss".format(split): cycleConLoss.detach().mean(),
                   "{}/recWHTR_loss".format(split): recWHTR_loss.detach().mean(),
                   "{}/HTR_loss".format(split): htr_loss.detach().mean(),
                   "{}/HTR_rec_loss".format(split): htr_rec_loss.detach().mean(),
                   "{}/cer".format(split): cer.detach().mean(),
                   "{}/acc".format(split): acc.detach().mean(),
                   "{}/prime_cer".format(split): cer_prime.detach().mean(),
                   "{}/prime_loss".format(split): primeLoss.detach().mean(),

                   #  "{}/adjusted_cycle_loss".format(split): adjusted_cycle_loss.detach().mean(),
                 #  "{}/adjusted_writer_loss".format(split): adjusted_writer_Loss.detach().mean(),
                   }
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
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