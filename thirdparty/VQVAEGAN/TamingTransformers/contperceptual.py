########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################
# originally from https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
#TODO proper cite

import torch
import torch.nn.functional as F

from thirdparty.VQVAEGAN.TamingTransformers.model import weights_init, NLayerDiscriminator


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss