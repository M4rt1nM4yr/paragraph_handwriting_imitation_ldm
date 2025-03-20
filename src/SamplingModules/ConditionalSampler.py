########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################
import argparse, os, sys, glob, datetime
import torch
import time
import numpy as np
from tqdm import trange
from thirdparty.VQVAEGAN.autoencoder import VQModel, VQModelInterfacePost, VQModelInterface, AutoencoderKL

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
from src.utils.utils import *


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
    if ckpt is None:
        model = LatentDiffusion()
    else:
        model = LatentDiffusion.load_from_checkpoint(ckpt,ld_ckpt_path=ckpt,strict=False)

    model.cuda()
    model.eval()
    return model

# so far we only take the style of the first sample in the batch but later on we should take more
def make_style_vector(from_example,batch_size=1,set="test",dl_apply=None,dl_file=None):


    if from_example:
        style_examples_dir = os.path.join(os.getcwd(),"StyleExamples")
        possible_style_examples = os.listdir(style_examples_dir)
        random_sample = possible_style_examples[np.random.randint(0,len(possible_style_examples))]
        img = Image.open(os.path.join(style_examples_dir,random_sample)).convert('L')
        img = (1.0 - torch.from_numpy(np.array(img))/255.0)-0.5
        pre_style = img.to('cuda').unsqueeze(dim=0)
        style_sample = torch.zeros((batch_size,1,pre_style.shape[1],pre_style.shape[2] ),device='cuda')
        for i in range(batch_size):
            style_sample[i] = pre_style

        style_padding = None

    else:

        gdm = instantiate_completely(dl_apply, dl_file,batch_size=1)

        if set =="test":
            testDL = gdm.test_dataloader()
        elif set=="val":
            testDL = gdm.val_dataloader()
        else:
            testDL = gdm.train_dataloader()

        iterator = iter(testDL)
        sample_count= torch.randint(0,len(testDL),(1,))[0]
        for i in range(sample_count):
            batch = next(iterator)
        #This is the style of the image we copy
        print("This is the writer ID: ",batch[WRITER])
        pre_style = batch[STYLE_SAMPLE].to('cuda')
        pre_padding = batch[STYLE_PADDING].to('cuda')

        style_sample = torch.zeros((batch_size,1,batch[IMAGE].shape[2],batch[IMAGE].shape[3] ),device='cuda')
        style_padding = torch.zeros((batch_size,pre_padding.shape[1],pre_padding.shape[2] ),device='cuda')

        for i in range(batch_size):
            style_padding[i] = pre_padding
            style_sample[i] = pre_style

    return style_sample, style_padding#batch[STYLE_PADDING].cuda()


def make_conditioning(str,batch_size = 1, style_sample=None,model=None,style_sample_padding=None):
    c = None
    pred_logits = None
    a = Alphabet()

    string_logits = a.string_to_logits(str).to('cuda')
    pred_logits = torch.ones((batch_size, len(str) +1), device='cuda')
    for i in range(batch_size):
        pred_logits[i][-1] = torch.tensor(3.0)
        pred_logits[i][0:-1] = string_logits

    pred_logits = pred_logits.type(torch.cuda.LongTensor)

    if style_sample_padding is None:
        style_sample_padding_adjusted = None
    else:
        style_sample_padding_adjusted = torch.zeros((batch_size,style_sample_padding.shape[1],style_sample_padding.shape[2]),dtype=torch.bool,device='cuda')

        for i in range(batch_size):
            style_sample_padding_adjusted[i] = style_sample_padding[0]



    if pa.use_conditioning == 2:

        logits = torch.ones((batch_size, len(str) + 2), device='cuda')
        for i in range(batch_size):
            logits[i][0] = torch.tensor(2.0)
            logits[i][-1] = 3
            logits[i][1:-1] = string_logits

        logits = logits.type(torch.cuda.LongTensor)
        tgt_mask = subsequent_mask(logits.shape[1]-1).to('cuda')
        #TODO check whether this works
        key_padding = create_padding_mask(logits)#torch.ones((1, logits.shape[1]), dtype=torch.bool, device='cuda')
        #key_padding[0, 0] = False

        con = (logits, tgt_mask, key_padding)
        c = model.get_learned_conditioning(con)

    if pa.use_conditioning == 3:
        cond_list = []
        for i in range(batch_size):
            cond_list.append(str)

        c = model.get_learned_conditioning(cond_list)

    if pa.use_conditioning == 4 or pa.use_conditioning == 5 or pa.use_conditioning == 6:
        logits = torch.ones((batch_size, len(str) + 2), device='cuda')
        for i in range(batch_size):
            logits[i][0] = torch.tensor(2.0)
            logits[i][-1] = 3
            logits[i][1:-1] = string_logits

        logits = logits.type(torch.cuda.LongTensor)
        tgt_mask = subsequent_mask(logits.shape[1]-1).to('cuda')
        # TODO check whether this works

        padding= create_padding_mask(logits)  # torch.ones((1, logits.shape[1]), dtype=torch.bool, device='cuda')
        # key_padding[0, 0] = False
        if model.cond_stage_concat_mode and pa.use_conditioning!= 5 and pa.use_conditioning!=6:
            style_padding = torch.zeros((batch_size,1), dtype=torch.bool,device='cuda')
            padding = torch.cat((padding,style_padding),dim=1)

        con = (logits, tgt_mask, padding, style_sample,style_sample_padding_adjusted)
        c = model.get_learned_conditioning(con)



    return c, pred_logits

def save_sampled_images(samples, save_path, base_count, timestep = None):
    #samples = torch.clamp(samples+0.5, min=0.0, max=1.0)
    #samples = torch.clamp( -1.0*(samples - 0.5), min=0.0, max=1.0)
    samples = torch.clamp( 1.0-(samples + 0.5), min=0.0, max=1.0)


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



def sample_run(logdir,model,sampler,c,uc,logits,batch_size,shape,base_count,scale,steps,eta):
    debug = True

    sample_path = os.path.join(logdir, "samples")
    intermediate_path = os.path.join(logdir, "pred_x0")
    xt_path = os.path.join(logdir, "x_T")

    os.makedirs(sample_path, exist_ok=True)

    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(xt_path, exist_ok=True)


    samples_ddim, intermediates = sampler.sample(S=steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 quantize_x0=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=eta,
                                                 logits=logits
                                                 #style_vector=style_vector,
                                                 #style_interpolation_scale=style_scale
                                                 )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    if debug:
        list_x_inter = intermediates['x_inter']
        list_predx0 = intermediates['pred_x0']
        time_steps = intermediates['index']
        for i in range(len(list_predx0)):
            predx0 = model.decode_first_stage(list_predx0[i])
            xinter = model.decode_first_stage(list_x_inter[i])
            save_sampled_images(predx0, intermediate_path, base_count=base_count, timestep=time_steps[i])
            save_sampled_images(xinter, xt_path, base_count=base_count, timestep=time_steps[i])

    base_count = save_sampled_images(x_samples_ddim, sample_path, base_count)
    samples_ddim.detach()

    return base_count




def start_sampling(batch_size,logdir,conditioning_string,diffusion_config,
                   set="test",remove_newline=False,guidanceScale=2.5,steps = 50,
                   dl_config = None,dl_apply=None,from_example=False,eta=0.0):


    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    ckpt = pa.ld_ckpt_path
   # model = load_model_start(ckpt)
    model = instantiate_completely("Diffusion\\ldm",diffusion_config)
    model.cuda()
    model.eval()

    print(75 * "=")
    print("logging to:")
    print(logdir)
    print(75 * "=")
    sampler = DDIMSampler(model)


    path = os.path.join(logdir,now)
    os.makedirs(path, exist_ok=True)

    style_sample, style_padding = make_style_vector(from_example=from_example,dl_apply=dl_apply,
                                                    dl_file=dl_config,batch_size=batch_size, set=set)
    empty_style = torch.zeros(style_sample.shape, device='cuda')-0.5

    dir_styles = "style_examples"
    style_path = os.path.join(path, dir_styles)
    os.makedirs(style_path, exist_ok=True)
    save_sampled_images(style_sample, style_path, base_count=-batch_size)


    if remove_newline:
        conditioning_string = conditioning_string.replace('\n', " ")

    eta = [eta]
    steps = [steps]
    scale = [guidanceScale]

    base_count = 0
    for i in range(len(steps)):
        #make directory for the test run
        print("parameter set ",i," of ",len(steps))
        name = "-s"+str(steps[i])+"-sc"+str(scale[i])+"-e"+str(eta[i])
        test_run_path = os.path.join(path, name)

        #make conditioning (besides style)
        with torch.no_grad():
            c, logits = make_conditioning(conditioning_string, batch_size=batch_size, style_sample=style_sample,
                                          style_sample_padding=style_padding,model=model)

            uc_style = empty_style
            uc, _ = make_conditioning("", batch_size=batch_size, style_sample=uc_style,model=model)

        shape = [model.model.diffusion_model.in_channels,
                 model.model.diffusion_model.image_size[0],
                 model.model.diffusion_model.image_size[1]]

        print("sample run: ","steps-",steps[i]," scale-",scale[i]," eta-",eta[i] )
        with torch.no_grad():
            base_count = sample_run(test_run_path, model, sampler, c, uc, logits, #style_vector,
                                  batch_size=batch_size, shape=shape, base_count=base_count, scale=scale[i], steps=steps[i],eta=eta[i])#,style_scale=style_interpolation_scales[i] )

