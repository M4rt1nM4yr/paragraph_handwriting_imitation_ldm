unet_conv_resample = True
unet_dims = 2
unet_num_classes = None
unet_use_checkpoint = False
unet_use_fp16 = False

unet_num_head = 1
unet_num_heads_channel = -1
unet_num_heads_upsample = -1
unet_use_scale_shift_norm = False
unet_resblock_updown = False

unet_use_new_attention_order = False
unet_use_spatial_transformer = True
unet_transformer_depth = 1

unet_use_pe_scaler = False
unet_n_embeded = None
unet_legacy = True

ld_lr = 0.00005#2.0e-06
ld_num_timesteps_cond = 1
ld_cond_stage_key = "text_logits_s2s"#"image"
ld_cond_stage_trainable = True
ld_concat_mode = True
ld_cond_stage_forward = None

# What Kind of Conditioning you want for the Diffusionprocess?
ld_conditioning_key='crossattn'# None # None, 'crossattn' , 'concat' , "class_label",
ld_scale_factor = 1.0
ld_scale_by_std = True
ld_ignore_key = []

cond_channels = 256
cond_hidden = 256
cond_ch_mult = (1,2,4,8,8)

ddpm_monitor = "val/loss_simple_ema"
ddpm_use_ema = True
ddpm_first_stage_key = "image"

ddpm_loss_type = "l2"
ddpm_ignore_keys = []
ddpm_load_only_unet = False
ddpm_log_every_t = 100
ddpm_clip_denoised = True

ddpm_orginal_elbo_weight = 0.
ddpm_v_posterior = 0.
ddpm_l_simple_weight = 1.
ddpm_conditioning_key = ld_conditioning_key #None #corssatn
ddpm_parameterization = "eps"

#this is actually not the pe we are gonna use...
ddpm_use_positional_encodings = False
ddpm_learn_logvar = False
ddpm_logvar_init=0.