"""
    This files defines most standard parameters and also the parameters the latent diffusion model is gonna work with.
    There are some presets you can use.

"""

import torch
from torch import nn
import numpy as np
#imports to debug
from PIL import Image
from PIL import ImageEnhance
from ParametersEncoder import *
from ParametersHTRandWriter import *
from ParametersDiffusion import *
import torchvision.transforms as T
from src.data.utils.alphabet import Alphabet
from PIL import ImageEnhance
from src.data.utils.constants import *



alpha = Alphabet()
back = T.ToPILImage()

#settings for all

"""
    IAM alone 672 writers
    IAM + synthetic data 1045 writers
"""
num_writers =1045# 672

size = (768,768)
z_shape = (96,96)


#TODO clean this section up. But be careful nothing breaks suddenly...

use_combined_dataloader = True
activate_style_padding = True

alphabet_size = 84
patience = 30
gpu_count = 1
scale = 1.0

# For training synthetic data
train_h5 = None# TODO?
val_h5 = None# TODO?
test_h5 = None# TODO?
use_synthetic_data = False


# settings for the cluster
only_Test = False
max_epochs = 50230 # 100
min_epochs = 1
batch_size = 8
n_workers = 8
persistent_workers = True
root = None#TODO IAM ROOT

#Writer Params
in_channels = embedded_dim

#HTR Params
#d_model = embedded_dim



"""
       PargraphMode 
       0: Paragraphen
       X > 0 : X-Zeilen
"""
ParagraphMode = 0


""""
    Unet-Model-Params
"""

unet_image_size = z_shape#(size[0]//(2**down_sample_steps),size[1]//(2**down_sample_steps))#(64)
unet_in_channels = embedded_dim
unet_out_channels = embedded_dim

unet_model_channels = 256
unet_context_dim = 1024
unet_channel_mult = (1, 2, 4)
unet_att_resolution = [4, 2, 1]

unet_num_res_blocks = 2
unet_dropout = 0.2

unet_use_pe_scaler = False

""" Latent Diffusion Parameters 

"""

""" Conditioning for latent Diffusion Parameters:
        CROSS_CONDITIONING 
"""
use_conditioning = CROSS_CONDITIONING

ld_use_conditioning = use_conditioning

ld_ckpt_path = None
ld_first_stage_ckpt = None

""" (DDPM) Parameters
"""

ddpm_timesteps = 1000
dppm_beta_scheduler = "linear"
ddpm_ckpt_path = None

ddpm_first_stage_ckpt = None
ddpm_image_size= unet_image_size#256
ddpm_channels = embedded_dim

ddpm_linear_start = 0.0002  # 1e-4
ddpm_linear_end = 0.0195
ddpm_cosine_s = 8e-3
ddpm_given_betas = None


ddpm_scheduler_config = None
ddpm_use_scheduler = True#True


"""
    3- AutoencoderKL
"""
ddpm_type_of_first_stage = AUTO_KL



"""
    Text and Style Conditioning Params
"""

cond_decoderLayerSet = True
affine_timestep_cond = False
pre_computed_writer = False#True
cond_concat_mode = False
cond_stage_dropout = 0.1
cond_stage_nr_encoders = 4
cond_extra_noise = False
remove_newline = False

# In case you wanna only reset one part of the network
reset_cond = False
reset_first_stage = False
reset_scheduler = False
reset_optimizers = False

writer_config = "768x768WriterCNN.yaml"

diffusion_dl_config = "general50k768x768.yaml"
dl_apply = "Dataloaders/768x768"
accumulate_grad_batches = 1

ch = 32
ch_mult = (1, 2, 4, 8)
ch_mult_htr = (1, 2, 4, 8)
checkpoint_htrw = None

embedded_dim = 1
z_channels = embedded_dim
d_model = 128

# cond stage
use_conditioning = CROSS_CONDITIONING
checkpoint_seq_writer = None

cond_ch_mult = (1, 2, 4, 8, 8)

cond_channels = 256
cond_hidden = 256

ld_cond_stage_trainable = True
z_shape_style_sample = (6, 6)


#augmentations for Diffusion probabilities

p_dilation =0.0
p_erosion = 0.0
p_distort_with_noise = 0.0
p_elastic_distortion = 0.0 # prev 0.0
p_random_transform = 0.0 #prev 0.0
p_perspective = 0.0
p_background = 0.0
