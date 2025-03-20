import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

"""
taken from: https://github.com/pytorch/vision/issues/908
"""
class FixedResizeStatic(object):
    def __init__(self, size, modeParagraph=0):
        if isinstance(size, int):
            size = (size,size)
        self.size = size
        self.mode = modeParagraph

    def __call__(self, img):

        #this is for synthetic data, so it's just an identity mapping
        if isinstance(img,np.ndarray):
            return img

        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)


        aspect_ratio_target = self.size[0] / self.size[1]
        aspect_ratio_input = img.height / img.width
        aspect_rev = self.size[1] / self.size[0]

        if aspect_ratio_input < aspect_ratio_target:
             # pad height

            aspect_rev = int(aspect_rev)
            final_width = img.width
            for i in range(aspect_rev):
                curren_width = final_width + i
                if curren_width % aspect_rev == 0:
                    final_width = curren_width
                    break

            final_height = int(aspect_ratio_target * final_width)
            out_img = Image.new(mode=img.mode, size=(final_width, final_height), color='white')
            out_img.paste(img)
            return F.resize(out_img, self.size)

        else:
            # pad width
            final_height = img.height
            final_width = int(final_height * aspect_rev+1)
            out_img = Image.new(mode=img.mode, size=(final_width, final_height), color='white')
            out_img.paste(img)
            return F.resize(out_img, self.size)




