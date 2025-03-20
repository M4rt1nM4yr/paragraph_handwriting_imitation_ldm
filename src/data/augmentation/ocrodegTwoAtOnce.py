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
from src.data.augmentation.ocrodeg import OcrodegAug

from src.data.augmentation.RandomTransform import RandomTransform
from src.data.augmentation.ElasticDistortion import ElasticDistortion

from torchvision.transforms import RandomPerspective
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import adjust_contrast, adjust_brightness



# data augmentation based on https://github.com/NVlabs/ocrodeg
class OcrodegAug2(OcrodegAug):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __call__(self, x):
        x_1 = np.array(x[0])
        x_2 = np.array(x[1])


        x_1 = np.array(x_1)
        x_1 = x_1 / (x_1.max() if x_1.max()>0 else 1)

        x_2 = np.array(x_2)
        x_2 = x_2 / (x_2.max() if x_2.max() > 0 else 1)

        pad_max = np.zeros(4)
        if self.p_random_vert_pad > torch.rand(1):
            pad_max[1] = x_1.shape[0] // 4
        if self.p_random_hori_pad > torch.rand(1):
            pad_max[3] = x_1.shape[0] * 2

        if np.sum(pad_max)>0:
            x_1 = ocrodeg.random_pad(x_1, border=pad_max)
            x_2 = ocrodeg.random_pad(x_2, border=pad_max)


        if self.p_random_squeeze_stretch > torch.rand(1):
            fx = np.random.uniform(low=0.8,high=1.2)
            x_1 = cv2.resize(x_1, None, fx=fx, fy=1, interpolation=cv2.INTER_LINEAR)
            x_2 = cv2.resize(x_2, None, fx=fx, fy=1, interpolation=cv2.INTER_LINEAR)

        if self.p_dilation > torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=2,high=4,size=(2,))))
            x_1 = torch.from_numpy(x_1).view(1,self.color_channels,x_1.shape[0],x_1.shape[1])
            x_2 = torch.from_numpy(x_2).view(1,self.color_channels,x_2.shape[0],x_2.shape[1])

            x_1 = morphology.erosion(x_1,kernel).squeeze().numpy()
            x_2 = morphology.erosion(x_2,kernel).squeeze().numpy()


        if self.p_erosion > torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=2,high=4,size=(2,))))
            x_1 = torch.from_numpy(x_1).view(1,self.color_channels,x_1.shape[0],x_1.shape[1])
            x_2 = torch.from_numpy(x_2).view(1,self.color_channels,x_2.shape[0],x_2.shape[1])

            x_1 = morphology.dilation(x_1,kernel).squeeze().numpy()
            x_2 = morphology.dilation(x_2,kernel).squeeze().numpy()

        for sigma in [2,5]:
            if self.p_distort_with_noise > torch.rand(1) and x_1.shape == x_2.shape:
                noise = ocrodeg.bounded_gaussian_noise(x_1.shape, sigma, 1.0)
                x_1 = ocrodeg.distort_with_noise(x_1, noise)

                noise2 = ocrodeg.bounded_gaussian_noise(x_2.shape, sigma, 1.0)

                x_2 = ocrodeg.distort_with_noise(x_2, noise2)


        x_1 = x_1 / (x_1.max() if x_1.max() > 0 else 1)
        x_2 = x_2 / (x_2.max() if x_2.max() > 0 else 1)

        x_1 = Image.fromarray((x_1*255).astype(np.uint8))
        x_2 = Image.fromarray((x_2*255).astype(np.uint8))

        if self.p_elastic_distortion > torch.rand(1):
            kernel = np.random.randint(2, 5)
            magnitude = np.random.randint(15, 20)
            x_1 = ElasticDistortion((kernel, kernel), (magnitude, magnitude), min_sep=(1, 1))(x_1)
            x_2 = ElasticDistortion((kernel, kernel), (magnitude, magnitude), min_sep=(1, 1))(x_2)



        if self.p_perspective > torch.rand(1):
            scale = np.random.uniform(0.0,0.15)
            x_1 = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(x_1)
            x_2 = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(x_2)

        if self.p_contrast > torch.rand(1):
            factor = np.random.uniform(0.9,0.99)
            x_1 = adjust_contrast(x_1, factor)
            x_2 = adjust_contrast(x_2, factor)

        if self.p_brightness > torch.rand(1):
            factor = np.random.uniform(0.95,
                                       0.99)
            x_1 = adjust_brightness(x_1, factor)
            x_2 = adjust_brightness(x_2, factor)

        return x_1, x_2


if __name__ == "__main__":
    a = OcrodegAug2(p_erosion=0.2)
    b = 1

