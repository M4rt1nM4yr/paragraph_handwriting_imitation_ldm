#HTR Params
n_head = 1
num_enc_layers = 2
num_dec_layers = 4
htr_lr = 0.0001
htr_dropout = 0.1
noisy_teacher = 0.2# 0.6
d_model = 128
encoder_dropout = 0.1
htrw_ch_mult = (1,2,4,8)
htr_ch = 16

#Writer Params. These might not be up to date anymore
hidden_size = 256
resolution = 512
writer_lr = 0.0001
writer_dropout = 0.0
seq_writer_ch_mult = (1,2,4,8,8)
seq_writer_ch = 16
checkpoint_seq_writer = None
resolution = 512

#TODO adjust this value
z_shape_style_sample = (6,6)