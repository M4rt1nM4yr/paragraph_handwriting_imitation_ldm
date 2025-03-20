import os.path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.augmentation.noTransform import NoTransform
from src.data.augmentation.noTransform import ShiftTransform

from src.data.dataset_fetcher import fetch_dataset
from src.data.utils.alphabet import Alphabet
from src.data.utils.FixedResizeStatic import FixedResizeStatic
from src.data.utils.constants import *

from src.data.utils.alphabet import Alphabet
import Parameters as pa
from src.model.modules.HTR_Writer import HTR_Writer
from src.data.utils.DataloaderUtils import *
from src.utils.utils import *
from src.data.augmentation.ocrodeg import OcrodegAug
import omegaconf
import PIL

class AbstractDataset(Dataset):
    def __init__(self,root_source=None, root_target=None,
                 dataset_type="IAM",
                 split="train", in_channels=1,
                 max_samples=-1,
                 size=pa.size,
                 z_shape = pa.z_shape,
                 modeParagraph=0,
                 style_samples = False,
                 z_shape_style = pa.z_shape_style_sample,
                 activate_style_padding=True,
                 remove_newline=False,
                 **kwargs):
        super(AbstractDataset, self).__init__()
        assert in_channels in [1,3]
        color_mode = "RGB" if in_channels==3 else "L"

        z_shape = tuple(z_shape)
        size = tuple(size)
        z_shape_style = tuple(z_shape_style)

        self.activate_style_padding = activate_style_padding
        self.alphabet = Alphabet(dataset="IAM", mode="both")
        self.z_shape_style = z_shape_style
        self.size = size
        self.z_shape = z_shape
        self.remove_newline = remove_newline
        self.mk_style_samples = style_samples
        self.modeParagraph = modeParagraph
        self.split = split

        self.sample_names, self.meta_data, self.images = fetch_dataset(root=root_source, dataset_type=dataset_type,
                                                                       mode=color_mode, split=split, size = size,
                                                                       max_samples=max_samples, alphabet=self.alphabet,
                                                                       modeParagraph=modeParagraph, **kwargs)
        print("fetched data")

        self.style_samples = None
        self.style_padding = None
        self.src_key_padding = dict()

        self.init_src_pad()
        self.init_style_dic()

        if remove_newline:
            nl = pa.alpha.string_to_logits('\n')[0]
            space = pa.alpha.string_to_logits(' ')[0]
            for name in self.sample_names:
                self.meta_data[name]["text"] = self.meta_data[name]["text"].replace('\n', " ")
                self.meta_data[name]["text_logits"][self.meta_data[name]["text_logits"] == nl] = space

                #to avoid random white spaces at the end
                if self.meta_data[name]["text"][-1] == " ":
                    self.meta_data[name]["text"] = self.meta_data[name]["text"][:-1]
                    self.meta_data[name]["text_logits"] = self.meta_data[name]["text_logits"][:-1]





    """
        Initializes all src pad mask
    """

    def init_src_pad(self):
        for name in self.sample_names:
            img = self.images[name]
            self.src_key_padding[name] = self.general_src_pad_mask(self.size,self.modeParagraph,self.z_shape,img)

    """
           Initializes all style samples and their mask if style_samples is set to true.
    """
    def init_style_dic(self):

        if not self.mk_style_samples:
            return

        size = self.size
        z_shape_style = self.z_shape_style
        modeParagraph = self.modeParagraph
        self.style_samples = dict()
        self.style_padding = dict()

        style_collection = dict()

        # first collect all the styles in a dictionary
        for s in self.sample_names:
            w = self.meta_data[s]["writer"]

            if style_collection.get(w) is None:
                style_collection[w] = list()
            style_collection[w].append(s)

            self.style_samples[s] = list()

        # assign style samples from the dictionary
        for s in self.sample_names:
            w = self.meta_data[s]["writer"]
            # if it is only one sample append that sample else remove the same style
            for style_sample in style_collection[w]:
                if style_sample == s and len(style_collection[w]) > 1:
                    continue
                self.style_samples[s].append(style_sample)

        for s in self.sample_names:
            current_style_image = self.images[s]
            self.style_padding[s] = self.general_src_pad_mask(size, modeParagraph, z_shape_style, current_style_image)


    """
        Creates src_pad_mask for a single real images or synthetic image.
        It determines the real/synthetic image based on data type of the image. np.ndarry being a synthetic image and 
        a PIL image being a real image.
        
        args:
            modeParagraph: How many lines does the image contain. 0 equals Paragraph image. 
                           Anything above 0 is not implemented yet cuz it's mainly used for pretraining.
            z_shape: The shape of the src pad mask
            image: The image we create the src pad mask for 
    """
    def general_src_pad_mask(self,size,modeParagraph,z_shape,image):
        if isinstance(image,np.ndarray):
            return mk_padding_interpolation(image,z_shape)
        if PIL.Image.isImageType(image):
            return single_src_pad_mask(size,modeParagraph,z_shape,image)

        return NotImplementedError


    """
        Creates src_pad_mask for a single real images only
        
        args:
            modeParagraph: How many lines does the image contain. 0 equals Paragraph image. 
                           Anything above 0 is not implemented yet cuz it's mainly used for pretraining.
            z_shape: Shape of the final src_pad mask
            image: PIL image in its original size 
    """

    def single_src_pad_mask(self,size, modeParagraph, z_shape, image):
        mask = torch.zeros(z_shape, dtype=torch.bool)

        # TODO adjust this later. It might already work with below implementation
        if modeParagraph > 0:
            return mask

        picture = image
        aspect_ratio_target = size[0] / size[1]
        aspect_ratio_input = picture.height / picture.width
        aspect_rev = int(size[1] / size[0])

        if aspect_ratio_input < aspect_ratio_target:
            # pad height

            final_width = picture.width
            for i in range(aspect_rev):
                curren_width = final_width + i
                if curren_width % aspect_rev == 0:
                    final_width = curren_width
                    break

            final_height = int(aspect_ratio_target * final_width)
        else:

            final_height = picture.height
            final_width = final_height * aspect_rev

        ratio_pad_height = picture.height / final_height
        ratio_pad_width = picture.width / final_width
        padding_start_height = min(int(ratio_pad_height * z_shape[0]) + 1, z_shape[0])
        padding_start_width = min(int(ratio_pad_width * z_shape[1]) + 1, z_shape[1])

        mask[:, padding_start_width:] = True
        mask[padding_start_height:, :] = True
        return mask

    """
            Creates src_pad_mask for a single synthetic image only

            args:
                x: image as a numpy array
                z_shape: shape of the final src_pad mask
                INTERPOLATION_CUTOFF : Constant to determine the cutoff for the padding
        """

    def mk_padding_interpolation(self,x, z_shape, INTERPOLATION_CUTOFF=0.99):
        # interpolation time
        padding = torch.zeros(z_shape, dtype=torch.bool)

        avg_horizontal = np.mean(x, axis=1) / 255  # torch.mean(x[0], dim=1)
        avg_vertical = np.mean(x, axis=0) / 255

        padding_start_horizontal = x.shape[0]
        padding_start_vertical = x.shape[1]

        # Check where the text starts for the height
        for j in range(x.shape[0] - 1, 0, -1):
            if avg_horizontal[j] < INTERPOLATION_CUTOFF:
                padding_start_horizontal = j
                break

        # Compute padding start for height
        padding_start_horizontal = min(padding_start_horizontal + 5, x.shape[0])
        ratio_pad_height = padding_start_horizontal / x.shape[0]
        padding_start_height = min(int(ratio_pad_height * z_shape[0]) + 1, z_shape[0])

        # Check where the text starts for the width
        for j in range(x.shape[1] - 1, 0, -1):
            if avg_vertical[j] < INTERPOLATION_CUTOFF:
                padding_start_vertical = j
                break

        # Compute the padding start for width
        padding_start_vertical = min(padding_start_vertical + 5, x.shape[1])
        ratio_pad_width = padding_start_vertical / x.shape[1]
        # Extra +1 here. So it doesn't accidentally delete single dots at the end of lines
        padding_start_width = min(int(ratio_pad_width * z_shape[1]) + 1 + 1, z_shape[1])

        padding[padding_start_height:, :] = True
        padding[:, padding_start_width:] = True

        return padding



    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, item):
        return NotImplementedError


class GeneralAugmentedDataset(AbstractDataset):
    def __init__(self,
                 augmentation=NoTransform(),
                 scale=1.0,
                 shift_scale=0.5,
                 **kwargs):

        super().__init__(**kwargs)

        if isinstance(augmentation,omegaconf.dictconfig.DictConfig):
            augmentation = instantiate_from_config(augmentation)


        resize = FixedResizeStatic(size=self.size,modeParagraph=self.modeParagraph)
        shift = ShiftTransform(shift_scale,scale)

        self.transform = transforms.Compose([
            augmentation,
            resize,
            transforms.ToTensor(),
            lambda x: 1 - x,
            shift,
        ])



    """
        
       :returns: 
                "name": Name of the sample
                IMAGE: Transformed image
                TEXT: Text depicted on the image
                WRITER: integer ID of the writer
                TEXT_LOGITS_CTC: text logits in form for CTC
                TEXT_LOGITS_S2S: text logits in form for s2s
                SRC_KEY_PADDING: Pad mask for the image
                STYLE_SAMPLE: The style sample transformed
                STYLE_PADDING: The src pad mask for the style image
       """

    def __getitem__(self, item):
        name = self.sample_names[item]

        style_sample = None
        style_padding = None

        if self.style_samples is not None:
            rng = torch.rand(1)[0]
            idx = int(rng * len(self.style_samples[name]))
            style_sample_name = self.style_samples[name][idx]
            style_sample = self.images[style_sample_name]
            style_sample = self.transform(style_sample)
            if self.activate_style_padding:
                #style_padding = self.style_padding[name][idx]
                style_padding = self.style_padding[style_sample_name]

        return {"name": name,
                IMAGE: self.transform(self.images[name]),
                TEXT: self.meta_data[name]["text"],
                WRITER: self.meta_data[name]["writer"],  # TODO see which way it is better#int(writer),
                TEXT_LOGITS_CTC: self.meta_data[name]["text_logits"],
                TEXT_LOGITS_S2S: torch.cat([torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                                            self.meta_data[name]["text_logits"],
                                            torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]])]),
                SRC_KEY_PADDING: self.src_key_padding[name],
                STYLE_SAMPLE: style_sample,
                STYLE_PADDING: style_padding,
                }


"""
    A special class that allows for Augmentations to be simultaneously applied to the style sample as well as the real 
    sample. There is no point to using this class besides for the Diffusion Model and only with ocrodegTwoAtOnce 
    Augmentations. Since style samples are necessary for this they are turned on mandatory. 

"""
from src.data.augmentation.ocrodegTwoAtOnce import OcrodegAug2


class GeneralSimultaneousAugmentationsDataset(AbstractDataset):
    def __init__(self,
                 augmentation=NoTransform(),
                 scale=1.0,
                 shift_scale=0.5,
                 label_dropout=0.0,
                 **kwargs):
        kwargs.pop('style_samples')
        super().__init__(style_samples=True,**kwargs)

        if self.split == "train":
            self.label_dropout= label_dropout
        else:
            self.label_dropout = 0.0

        if isinstance(augmentation, omegaconf.dictconfig.DictConfig):
            augmentation = instantiate_from_config(augmentation)



        resize = FixedResizeStatic(size=self.size, modeParagraph=self.modeParagraph)
        shift = ShiftTransform(shift_scale, scale)
        self.combined_transform = transforms.Compose([
            augmentation,

        ])

        self.transform = transforms.Compose([
            resize,
            transforms.ToTensor(),
            lambda x: 1 - x,
            shift,
        ])
    """

       :returns: 
                "name": Name of the sample
                IMAGE: Transformed image
                TEXT: Text depicted on the image
                WRITER: integer ID of the writer
                TEXT_LOGITS_CTC: text logits in form for CTC
                TEXT_LOGITS_S2S: text logits in form for s2s
                SRC_KEY_PADDING: Pad mask for the image
                STYLE_SAMPLE: The style sample transformed
                STYLE_PADDING: The src pad mask for the style image
       """

    def __getitem__(self, item):
        name = self.sample_names[item]

        style_padding = None
        rng = torch.rand(1)[0]
        idx = int(rng * len(self.style_samples[name]))

        style_sample_name = self.style_samples[name][idx]
        style_sample = self.images[style_sample_name]

        combined = (self.images[name],style_sample)#,dim=0)
        img,style_sample = self.combined_transform(combined)

        if self.activate_style_padding:
            style_padding = self.style_padding[style_sample_name]


        if torch.rand(1) < self.label_dropout:
            x = self.transform(img)
            return {"name": name,
                    IMAGE: x,
                    TEXT: "",
                    WRITER: self.meta_data[name]["writer"],  # TODO see which way it is better#int(writer),
                    TEXT_LOGITS_CTC: pa.alpha.string_to_logits(""),
                    TEXT_LOGITS_S2S: torch.cat([torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                                                pa.alpha.string_to_logits(""),
                                                torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]])]),
                    SRC_KEY_PADDING: self.src_key_padding[name],
                    STYLE_SAMPLE: torch.zeros(x.shape)-0.5,
                    STYLE_PADDING: torch.zeros(self.z_shape_style).type(torch.bool),
                    }



        return {"name": name,
                IMAGE: self.transform(img),
                TEXT: self.meta_data[name]["text"],
                WRITER: self.meta_data[name]["writer"],  # TODO see which way it is better#int(writer),
                TEXT_LOGITS_CTC: self.meta_data[name]["text_logits"],
                TEXT_LOGITS_S2S: torch.cat([torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                                            self.meta_data[name]["text_logits"],
                                            torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]])]),
                SRC_KEY_PADDING: self.src_key_padding[name],
                STYLE_SAMPLE: self.transform(style_sample),
                STYLE_PADDING: style_padding,
                }



