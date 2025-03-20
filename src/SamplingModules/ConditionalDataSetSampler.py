########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################

import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
import uuid

#from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageOps

from src.diffusion.ddim import DDIMSampler

import Parameters as pa
from src.diffusion.ddpm import LatentDiffusion
from src.data.utils.alphabet import Alphabet
from src.data.utils.constants import *
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from src.model.modules.HTR_Writer import HTR_Writer
from src.data.augmentation.ocrodeg import OcrodegAug
from src.model.modules.WriterSequence import WriterSequence
import h5py
from src.utils.utils import *

#from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.



def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
#    x = x.permute(1, 2, 0).numpy()
    x = x.numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
#    if not x.mode == "RGB":
 #       x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, cond,shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(cond, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = ~(torch.from_numpy(subsequent_mask) == 0).squeeze(0)
    matrix_ninf = torch.ones(()) * float('-inf')
    matrix_zeros = torch.zeros(()).float()
    subsequent_mask = torch.where(subsequent_mask,matrix_ninf,matrix_zeros)
    return subsequent_mask


#TODO make this function work for more than 1 string at once
def single_string_to_conditioning_informations(cond):
    alph = Alphabet()
    logits = alph.string_to_logits(cond).to('cuda')
    tgt_mask = subsequent_mask(logits.shape[0]-1).to('cuda')
    tgt_key_padding = torch.zeros((1,logits.shape[0]-1), dtype=torch.bool,device='cuda')

    logits = logits.reshape((1,logits.shape[0]))[:,:-1]

    return (logits,tgt_mask,tgt_key_padding)
@torch.no_grad()
def convsample_ddim(model,cond, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]

    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model,cond_batch, batch_size,  vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size[0],
             model.model.diffusion_model.image_size[1]]
    c = model.get_learned_conditioning(cond_batch)

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model,c, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,c,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, cond_batch, batch_size=2, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None,scale=1.0,uc=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    all_images = []
    print(f"Running conditional sampling for {n_samples} samples")


    for _ in trange(n_samples // batch_size, desc="Sampling Batches (conditional)"):
        logs = make_convolutional_sample(model,cond_batch, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
        n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
        all_images.extend([custom_to_np(logs["sample"])])
        if n_saved >= n_samples:
            print(f'Finish after generating {n_saved} samples')
            break
    all_img = np.concatenate(all_images, axis=0)
    all_img = all_img[:n_samples]
    shape_str = "x".join([str(x) for x in all_img.shape])
    nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
    np.savez(nppath, all_img)



    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x[0])
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def create_padding_mask(tgt):
    return torch.eq(tgt,torch.ones(tgt.shape, dtype=torch.long,device='cuda') * torch.LongTensor([1]).to('cuda'))

def load_model_start(ckpt):
    model = load_model_(ckpt)

    return model

def load_model_(ckpt):
    model = LatentDiffusion.load_from_checkpoint(ckpt,ld_ckpt_path=ckpt,strict=False)

    model.cuda()
    model.eval()
    return model

def make_conditioning(c,model):
    logits,tgt_mask,tgt_pad_mask,style_sample, padding_mask = c

    if pa.use_conditioning == 2:
        con = (logits, tgt_mask, tgt_pad_mask)

    if pa.use_conditioning == 4 or pa.use_conditioning == 5 or pa.use_conditioning == 6:

        if model.cond_stage_concat_mode and tgt_pad_mask!=None and pa.use_conditioning!= 5 and pa.use_conditioning!=6:
            style_padding = torch.zeros((tgt_pad_mask.shape[0],1), dtype=torch.bool,device='cuda')
            tgt_pad_mask = torch.cat((tgt_pad_mask,style_padding),dim=1)

        con = (logits, tgt_mask, tgt_pad_mask, style_sample,padding_mask )

    c = model.get_learned_conditioning(con)

    return c

def save_sampled_images(samples, save_path, base_count, timestep = None):
    #samples = torch.clamp(samples+0.5, min=0.0, max=1.0)
    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in samples:
        x_sample = 255. * x_sample.cpu().numpy()
        # ImageOps.invert(Image.fromarray(x_sample.astype(np.uint8)[0] )).save(os.path.join(sample_path, f"{base_count:04}.png"))
        filename = str(base_count)
        if timestep is not None:
            filename =filename + "-t-"+str(timestep)
        filename = filename+".png"
        Image.fromarray(x_sample.astype(np.uint8)[0]).save(os.path.join(save_path, filename))


        base_count += 1
    return base_count

def post_process_sampled_images(x, method = 0):
    if method == 0:
        return 1.0-torch.clamp(x+0.5, min=0.0, max=1.0)

    elif method == 1:
        return 1.0-torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    elif method ==2:
        return 1.0-torch.clamp(-1.0*(x - 0.5), min=0.0, max=1.0)

    else:
        print("Method not implemented")
        return None

def sample_run(model,sampler,c,uc,batch_size,shape,base_count,scale,steps,eta,style_scale,logits=None,style_vector=None):
    samples_ddim, intermediates = sampler.sample(S=steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 quantize_x0=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=eta,
                                                 logits=logits,
                                                 style_vector=style_vector
                                                 )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    return x_samples_ddim,base_count+len(samples_ddim)

def get_iterator_over_data_module(batch_size,dl_apply=None,dl_config=None):
    """
    def get_iterator_over_data_module(data_type,batch_size, paraMode=pa.ParagraphMode, cluster=False,
                                      h5= None, root_source="",remove_newline=False,activate_style_padding=True,
                                      dl_apply=None,dl_config=None):
    """
    aug = OcrodegAug()
    gdm = instantiate_completely(dl_apply,dl_config,batch_size=batch_size,augmentation=aug)

    testDL = gdm.test_dataloader()
    iterator = iter(testDL)
    data_size = len(testDL.dataset)

    return iterator, data_size


def string_to_logits(str,batch_size):
    string_logits = pa.alpha.string_to_logits(str).to('cuda')
    logits = torch.ones((batch_size, len(str) + 2), device='cuda')
    for i in range(batch_size):
        logits[i][-1] = torch.tensor(3.0)
        logits[i][1:-1] = string_logits
        logits[i][0] = torch.tensor(2.0)

    logits = logits.type(torch.cuda.LongTensor).cuda()
    return logits



def get_conditioning(batch,datatype,batch_size=1,size=pa.size):

    c_in = None
    c_sample = None

    if datatype == 0 or datatype == 1:
        #information for the Diffusion process
        text = batch[TEXT_LOGITS_S2S][0].cuda().repeat(batch_size).reshape((batch_size,batch[TEXT_LOGITS_S2S].shape[1]))
        style_sample = batch[STYLE_SAMPLE][0].cuda().repeat((batch_size,1,1)).reshape((batch_size,batch[STYLE_SAMPLE].shape[1],
                                                                                 batch[STYLE_SAMPLE].shape[2],
                                                                                 batch[STYLE_SAMPLE].shape[3]))
        tgt_mask = batch[TGT_MASK].cuda().repeat((batch_size,1,1)).reshape((batch_size,batch[TGT_MASK].shape[0],
                                                                            batch[TGT_MASK].shape[1]))
        tgt_key_padding = batch[TGT_KEY_PADDING_MASK][0].cuda().repeat((batch_size,1)).reshape((batch_size,
                                                                                                batch[TGT_KEY_PADDING_MASK].shape[1]))
        style_padding = batch[STYLE_PADDING][0].cuda().repeat((batch_size,1,1)).reshape((batch_size,
                                                                                         batch[STYLE_PADDING].shape[1],
                                                                                         batch[STYLE_PADDING].shape[2]))

        c_in = (text,tgt_mask,tgt_key_padding,style_sample,style_padding)


        #Information for the sample we are generating
        clean_text = batch[TEXT_LOGITS_CTC].cuda()
        writer = batch[WRITER].cuda()
        comparasion_sample = batch[IMAGE].cuda()
        style_sample = batch[STYLE_SAMPLE].cuda()
        name = batch["name"]

        c_sample = (clean_text,writer,comparasion_sample,style_sample,name)

    #empty case. Required for Classifier-Free-Guidance
    elif datatype == -1:
        size = batch[STYLE_SAMPLE].shape
        empty_text = string_to_logits("",batch_size).cuda()
        empty_style = torch.zeros((batch_size,size[1],size[2],size[3]),device='cuda')-0.5 #-0.5
        # try masking it out completely a = torch.ones(10, dtype=torch.bool)
        c_in = (empty_text,None,None,empty_style,None)

        c_sample = None
    else:
        print("Error no implemented")

    return c_in, c_sample

def make_style_vector(batch, sampler):
    style_sample = batch[STYLE_SAMPLE].cuda()
    if sampler.writer is not None:
        style_vector = sampler.writer(style_sample)
    else:
        _,_ , _, style_vector = sampler.htrw(batch[TEXT_LOGITS_S2S][:,:-1].cuda(),memory=style_sample,s_mask=batch[TGT_MASK].cuda(),
                                    tgt_key_padding_mask= batch[TGT_KEY_PADDING_MASK][:,:-1].cuda())


    return style_vector


"""
    Method to sample from the dataset. If batch_size = 1, the resulting h5 file can be directly used with the synthetic dataloader.
    Otherwise you have to select the resulting images.

"""
def start_sampling(diffusion_config,batch_size,logdir,data_type=1,sampling_hyper_parameters = (450,1.5,0.0,0.0),
                   description = "",dl_config=None,dl_apply=None ):

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    now = now
    if description != "":
        now = now+"-"+description# + "-"+pa.dppm_beta_scheduler+"-set-"+
    sys.path.append(os.getcwd())

    model = instantiate_completely("Diffusion\\ldm", diffusion_config)
    model.cuda()
    model.eval()

    print(75 * "=")
    print("logging to:")
    print(logdir)
    print(75 * "=")

    print("---Using classifier free guidance---")
    sampler = DDIMSampler(model)

    path = os.path.join(logdir,now)
    os.makedirs(path, exist_ok=True)

    iter, amount_of_samples = get_iterator_over_data_module(batch_size=1,dl_config=dl_config, dl_apply=dl_apply)

    base_count = 0
    steps = sampling_hyper_parameters[0]
    scale = sampling_hyper_parameters[1]
    eta = sampling_hyper_parameters[2]
    style_guidance_extra_scale = sampling_hyper_parameters[3] # Don't use

    shape = [model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size[0],
            model.model.diffusion_model.image_size[1]]


    with h5py.File(os.path.join(path, "x.h5"), "w") as synthetic_data:
        total_data_points = amount_of_samples   # found by running the algorithm on all data points in the training data
        image_shape = (total_data_points,)

        # vll int8 ausprobieren fuer mehr Daten
        h5_image = synthetic_data.create_dataset(
            "images", (total_data_points,), dtype=h5py.vlen_dtype(np.dtype("uint8")), chunks=True
        )
        h5_label = synthetic_data.create_dataset(
            "labels",
            (total_data_points,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        h5_id = synthetic_data.create_dataset(
            "id",
            (total_data_points,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        h5_writer = synthetic_data.create_dataset(
            "writer",
            (total_data_points,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        h5_original = synthetic_data.create_dataset(
            "original",
            image_shape,
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
            chunks=True,
        )
        h5_style_sample = synthetic_data.create_dataset(
            "style_sample",
            image_shape,
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
            chunks=True,
        )
        h5_name = synthetic_data.create_dataset(
            "name",
            (total_data_points,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        h5_batch_size = synthetic_data.create_dataset(
            "batch_size",
            (total_data_points,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )

        j = 0
        counter = -1

        for batch in iter :
            print(counter, " of ", total_data_points)
            style_vector = None
            uc = None

            c_in, c_sample = get_conditioning(batch,datatype=data_type)
            with torch.no_grad():
                c = make_conditioning(c_in,model)

                c_in_empty, _ = get_conditioning(batch, datatype=-1)
                uc = make_conditioning(c_in_empty, model)
                #run
                sampled_images, base_count = sample_run(model, sampler, c, uc,
                                        batch_size=batch_size, shape=shape, base_count=base_count, scale=scale,
                                        steps=steps, eta=eta, style_scale=style_guidance_extra_scale,logits=c_in[0],
                                                        style_vector=style_vector)
            #save
            s_images = np.zeros((batch_size,sampled_images.shape[2],sampled_images.shape[3]))

            i = 0
            for img in sampled_images:
                s_images[i] = np.asarray(pa.back(post_process_sampled_images(img)[0]).convert("L"))
                i = i+1

            counter = counter + 1

            try:
                text_uuid = str(uuid.uuid4())  # str(uuid.uuid4())
                label = pa.alpha.logits_to_string(c_sample[0][0])

                h5_id[counter] = text_uuid
                h5_image[counter] = s_images.reshape(1, -1)#np.asarray(cropped_image).reshape((batch_size, -1))
                h5_label[counter] = label
                h5_writer[counter] = str(c_sample[1][0].cpu().item())  # c_sample[1][counter].cpu()
                h5_original[counter] = np.asarray(pa.back(1.0 - (c_sample[2][0][0] + 0.5)).convert("L")).reshape(
                    (1, -1))  # c_sample[2][counter]
                h5_style_sample[counter] = np.asarray(pa.back(1.0 - (c_sample[3][0][0] + 0.5) ).convert("L")).reshape(
                    (1, -1))
                h5_name[counter] = c_sample[4][0]

                h5_batch_size[counter] = str(batch_size)

            except Exception as e:
                print(e)
                j = j + 1
                print(f"Error happened {j} times")


