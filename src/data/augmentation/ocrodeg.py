#TODO citation?
import random

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from kornia import morphology

from thirdparty import ocrodeg
from PIL import Image

from src.data.augmentation.RandomTransform import RandomTransform
from src.data.augmentation.ElasticDistortion import ElasticDistortion

from torchvision.transforms import RandomPerspective
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import adjust_contrast, adjust_brightness



# data augmentation based on https://github.com/NVlabs/ocrodeg
class OcrodegAug(nn.Module):
    def __init__(self,
                 p_random_vert_pad=0.,
                 p_random_hori_pad=0.,
                 p_random_squeeze_stretch=0.,
                 p_dilation=0.,
                 p_erosion=0.,
                 p_distort_with_noise=0.,
                 p_background_noise=0.,
                 p_elastic_distortion = 0.,
                 p_random_transform = 0.,
                 p_perspective = 0.,
                 p_contrast=0.,
                 p_brightness=0.,
                 color_channels = 1):
        super(OcrodegAug, self).__init__()
        self.p_random_vert_pad = p_random_vert_pad
        self.p_random_hori_pad = p_random_hori_pad
        self.p_random_squeeze_stretch = p_random_squeeze_stretch
        self.p_dilation = p_dilation
        self.p_erosion = p_erosion
        self.p_distort_with_noise = p_distort_with_noise
        self.p_background_noise = p_background_noise
        self.noise_bg = ocrodeg.FastPrintlike()

        self.toTensor = transforms.ToTensor()
        self.color_channels = color_channels

        self.p_elastic_distortion = p_elastic_distortion
        self.p_random_transform = p_random_transform

        self.p_perspective = p_perspective
        self.p_contrast = p_contrast
        self.p_brightness = p_brightness

    def __call__(self, x):
        x = np.array(x)
        x = x / (x.max() if x.max()>0 else 1)
        pad_max = np.zeros(4)
        if self.p_random_vert_pad > torch.rand(1):
            pad_max[1] = x.shape[0] // 4
        if self.p_random_hori_pad > torch.rand(1):
            pad_max[3] = x.shape[0] * 2
        if np.sum(pad_max)>0:
            x = ocrodeg.random_pad(x, border=pad_max)

        if self.p_random_squeeze_stretch > torch.rand(1):
            fx = np.random.uniform(low=0.8,high=1.2)
            x = cv2.resize(x, None, fx=fx, fy=1, interpolation=cv2.INTER_LINEAR)

        if self.p_dilation > torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=2,high=5,size=(2,))))
            x = torch.from_numpy(x).view(1,self.color_channels,x.shape[0],x.shape[1])
            x = morphology.erosion(x,kernel).squeeze().numpy()

        if self.p_erosion > torch.rand(1):
            kernel = torch.ones(2,2)
            x = torch.from_numpy(x).view(1,self.color_channels,x.shape[0],x.shape[1])
            x = morphology.dilation(x,kernel).squeeze().numpy()

        for sigma in [2,5]:
            if self.p_distort_with_noise > torch.rand(1):
                noise = ocrodeg.bounded_gaussian_noise(x.shape, sigma, 3.0)
                x = ocrodeg.distort_with_noise(x, noise)

        x = x / (x.max() if x.max() > 0 else 1)
        if self.p_background_noise > torch.rand(1):
            x = 1-self.noise_bg(x)

        x = Image.fromarray((x*255).astype(np.uint8))

        if self.p_elastic_distortion > torch.rand(1):
            kernel = np.random.randint(1, 3)
            magnitude = np.random.randint(1, 8)
            x = ElasticDistortion((kernel, kernel), (magnitude, magnitude), min_sep=(1, 1))(x)

        if self.p_random_transform > torch.rand(1):
            x = RandomTransform(8)(x)

        if self.p_perspective > torch.rand(1):
            scale = np.random.uniform(0.0,0.15)
            x = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(x)

        if self.p_contrast > torch.rand(1):
            factor = np.random.uniform(0.2,0.4)
            x= adjust_contrast(x, factor)

        if self.p_brightness > torch.rand(1):
            factor = np.random.uniform(0.01,
                                       0.5)
            x = adjust_brightness(x, factor)

        return x




