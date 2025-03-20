#settings for the Encoder decoder
attention_resolution = []# [64,32]# [16]
temb = 0
ch = 32
ch_mult = (1,2,4,8)
ch_mult_htr = (1,2,4,8)


double_z = False
auto_used = False
kl_weight = 0.000001

checkpoint_htrw = None
checkpoint_Writer = None
checkpoint_HTR = None

checkpoint_VQVAE = None
ckpt_autoKL = None


embedded_dim = 1# 256#128#256
z_channels = embedded_dim#256

vq_vae_dropout = 0.1
quantizer_legacy = False


htr_weight = 0.3
writer_weight = 0.005#1
disc_weight = 0.5

disc_start = 9980001
n_embedded = 1024#8192
noisy_teacher_vqvae = 0.0

""" Lambda Scheduler Params (LR Diffusion)

"""
#TODO are these good values?
lr_warm_up_steps = [10000]
lr_cycle_lengths = [10000000000000]
lr_start = [1.e-6]
lr_max = [1.]
lr_min = [1.]

""" Lambda Scheduler Params (Transformers)

"""
#TODO are these good values?
lam_warm_up_steps = [10000]
lam_cycle_lengths = [10000000000000]
lam_f_start = [1.e-6]
lam_f_max = [1.]
lam_f_min = [1.]
