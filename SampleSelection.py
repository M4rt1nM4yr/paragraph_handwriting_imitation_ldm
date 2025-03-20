import os
import string
import sys,  glob, datetime

import torch
from torch import nn
from torchvision import transforms
import pytorch_lightning as pl

from src.data.augmentation.noTransform import NoTransform
from src.data.dataset_fetcher import fetch_dataset
from src.data.utils.alphabet import Alphabet
from src.data.utils.constants import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Parameters import *

from src.data.augmentation.ocrodeg import OcrodegAug

from torch.utils.data import DataLoader
from src.data.utils.custom_collate import custom_collate

# from pytorch_lightning.profiler import SimpleProfiler

from HTRTrainer import HTR_Writer
from src.utils.utils import *
from src.data.augmentation.noTransform import NoTransform
from src.data.augmentation.noTransform import ShiftTransform
import h5py
import uuid
import torchmetrics
from tqdm import tqdm
from thirdparty.WriterSelection.sift_vlad import WriterSelect

from argparse import ArgumentParser





cer = torchmetrics.CharErrorRate()
shift = ShiftTransform(0.5,scale)
alphabet = Alphabet(dataset="all")

transform2 = transforms.Compose([
        transforms.ToTensor(),
        lambda x: 1 - x,
        shift,

])


def mk_padding_interpolation(x,z_shape, INTERPOLATION_CUTOFF = 0.99,invert=False):
    # interpolation time
    padding = torch.zeros(z_shape, dtype=torch.bool)

    if invert:
        avg_horizontal =255- np.mean(x, axis=1) / 255  # torch.mean(x[0], dim=1)
        avg_vertical =255- np.mean(x, axis=0) / 255
    else:
        avg_horizontal = np.mean(x,axis=1)/255#torch.mean(x[0], dim=1)
        avg_vertical = np.mean(x, axis=0) / 255

    padding_start_horizontal = x.shape[0]
    padding_start_vertical = x.shape[1]

    #Check where the text starts for the height
    for j in range(x.shape[0] - 1, 0, -1):
        if avg_horizontal[j] < INTERPOLATION_CUTOFF:
            padding_start_horizontal = j
            break

    #Compute padding start for height
    padding_start_horizontal = min(padding_start_horizontal + 5, x.shape[0])
    ratio_pad_height = padding_start_horizontal / x.shape[0]
    padding_start_height = min(int(ratio_pad_height * z_shape[0]) + 1,z_shape[0])


    #Check where the text starts for the width
    for j in range(x.shape[1] - 1, 0, -1):
        if avg_vertical[j] < INTERPOLATION_CUTOFF:
            padding_start_vertical = j
            break

    #Compute the padding start for width
    padding_start_vertical = min(padding_start_vertical + 5, x.shape[1])
    ratio_pad_width = padding_start_vertical / x.shape[1]
    # Extra +1 here. So it doesn't accidentally delete single dots at the end of lines
    padding_start_width= min(int(ratio_pad_width * z_shape[1]) + 1 +1,z_shape[1])

    padding[padding_start_height:, :] = True
    padding[:,padding_start_width:] = True

    return padding


def get_htr_score(htr_model,img,params,src_padding_mask):

    EOS = torch.tensor(3)
    img = htr_model.encode(transform2(img[0]).unsqueeze(dim=0).cuda())
    pred, predicted_logits = htr_model.inference_htr(img,max_char_len= params[TEXT_LOGITS_S2S].shape[0],
                                                     memory_key_padding_mask=src_padding_mask.unsqueeze(dim=0).cuda())
    predicted_characters = alphabet.batch_logits_to_string_list(predicted_logits, [EOS])
    correct_characters = alphabet.batch_logits_to_string_list(params[TEXT_LOGITS_S2S].unsqueeze(dim=0)[:, 1:], [EOS])
    htr_score = cer(predicted_characters[0], correct_characters[0])

    return htr_score

def get_writer_score(writer_model,img,params,src_padding_mask):
    writer_rankings, dist_matrix = writer_model.tests(img, params[STYLE_SAMPLE],
                 params_dict[names[0]][WRITER], prepare_data=True)


    return writer_rankings, dist_matrix

def get_combined_score(htr_rankings, writer_rankings,cer_importance=1.0001,writer_importance=1.0):

    max_score = len(htr_rankings)
    ranking = np.zeros((max_score))

    current_score = max_score
    for i in range(len(htr_rankings)):
        ranking[htr_rankings[i]] = ranking[htr_rankings[i]] + current_score *cer_importance
        ranking[writer_rankings[i]] = ranking[writer_rankings[i]] + current_score *writer_importance
        current_score = current_score -1

    return ranking.argmax()



"""
    transforms the datasets into a dictionary
    img_dict:  name -> images
    params_dict: name -> parameters
    names: list of all sample names
"""

def get_datasets(file_paths,size,remove_newline):

    img_dict = dict()
    params_dict = dict()
    names = []

    for file in file_paths:
        dataset = h5py.File(file, "r")
        length = len(dataset["writer"])


        for i in range(length):
            batch_size = 10
            if dataset.get("batch_size") is not None:
                batch_size = int(dataset["batch_size"][i].decode("utf-8"))

            #Go through images and add them to the dicionary based on the name
            images = dataset["images"][i].reshape(batch_size,1,size[0],size[1])

            if dataset.get("name") is None:
                print("careful names and ids might swap")
                name = str(i)
            else:
                name = dataset["name"][i].decode("utf-8")

            if img_dict.get(name) is None:
                img_dict[name] = list()

            # So we add all the sampled images to the same batch
            for img in images:
                img_dict[name].append(img)

            if params_dict.get(name) is None:
                text = dataset["labels"][i].decode("utf-8")  # .replace("\n", " ")
                if remove_newline:
                    text = text.replace('\n', " ")
                logits = alphabet.string_to_logits(text)

                if dataset.get("style_sample") is not None:
                    style_sample = dataset["style_sample"][i]
                else:
                    style_sample = torch.zeros(size)

                params_dict[name] = {
                    STYLE_SAMPLE: style_sample.reshape((1,size[0],size[1])),
                    ORIGINAL: dataset["original"][i].reshape((1,size[0],size[1])),
                    TEXT: text,
                    WRITER: int(dataset["writer"][i].decode("utf-8")),
                    TEXT_LOGITS_S2S: torch.cat([torch.LongTensor([alphabet.toPosition[START_OF_SEQUENCE]]),
                                                logits,
                                                torch.LongTensor([alphabet.toPosition[END_OF_SEQUENCE]])]),
                    "name": name
                }
                names.append(name)
    return img_dict, params_dict,names



def get_full_ranking(img_dict, params_dict,names,htr_model,writer_model,z_shape,size,save_path,note,nr_of_samples_per_batch=10):

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(save_path,"Full_ranking")
    os.makedirs(save_path,exist_ok=True)

    with h5py.File(os.path.join(save_path, note+now+"_full_ranking_x.h5"), "w") as synthetic_data:
        total_data_points = len(names)   # found by running the algorithm on all data points in the training data
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
        h5_writer_score = synthetic_data.create_dataset(
            "writer_score",
            image_shape,
            dtype=h5py.vlen_dtype(np.dtype("float32")),
            chunks=True,
        )
        h5_cer = synthetic_data.create_dataset(
            "cer_score",
            (total_data_points,),
            dtype=h5py.vlen_dtype(np.dtype("float32")),
            chunks=True,
        )



        counter = 0
        for name in tqdm(names):
            params = params_dict[name]
            images = img_dict[name]

            htr_cers = []

           # print("----- sample: ",name)
            for img in images:
                src_padding_mask = mk_padding_interpolation(img[0],z_shape)
                htr_cers.append(get_htr_score(htr_model,img,params,src_padding_mask).item())

            writer_ranking, dist_matrix = get_writer_score(writer_model,images,params,src_padding_mask)

            print("CERS: ",htr_cers)
            print(" SIMILARITIES ",dist_matrix)
            img_collection = np.zeros((nr_of_samples_per_batch,size[0],size[1]))
            for j in range(len(images)):
                img_collection[j] = images[j][0]

            label = params[TEXT]
            text_uuid = str(uuid.uuid4())  # str(uuid.uuid4())

            h5_id[counter] = text_uuid
            h5_image[counter] = img_collection.reshape((1, - 1))  # np.asarray(cropped_image).reshape((batch_size, -1))
            h5_label[counter] = label
            h5_writer[counter] = str(params[WRITER])  # c_sample[1][counter].cpu()
            h5_original[counter] = params[ORIGINAL] .reshape((1, - 1)) # c_sample[2][counter]
            h5_style_sample[counter] = params[STYLE_SAMPLE].reshape((1, - 1))
            h5_name[counter] = str(name)
            h5_cer[counter] = np.array(htr_cers)
            h5_writer_score[counter] = np.array(dist_matrix)

            counter = counter +1


def get_best_ranking(img_dict, params_dict,names,htr_model,writer_model,z_shape,size,save_path,note,cer_importance,writer_importance):
    save_path = os.path.join(save_path,"Filtered")
    os.makedirs(save_path,exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with h5py.File(os.path.join(save_path, note+now+"_filtered_x.h5"), "w") as synthetic_data:
        total_data_points = len(names)   # found by running the algorithm on all data points in the training data
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

        counter = 0
        for name in tqdm(names):
            params = params_dict[name]
            images = img_dict[name]

            htr_cers = []

           # print("----- sample: ",name)
            for img in images:
                src_padding_mask = mk_padding_interpolation(img[0],z_shape)
                if cer_importance == 0.0:
                    htr_cers.append(0.0)
                else:
                    htr_cers.append(get_htr_score(htr_model,img,params,src_padding_mask).item())


            cer_ranking = np.array(htr_cers).argsort()
            if writer_importance == 0.0:
                writer_ranking = np.arange(len(images))#,device='cuda')
            else:
                writer_ranking,_ = get_writer_score(writer_model,images,params,src_padding_mask)

            best_idx = get_combined_score(cer_ranking,writer_ranking,cer_importance,writer_importance)
            best_image = images[best_idx]

            label = params[TEXT]
            text_uuid = str(uuid.uuid4())  # str(uuid.uuid4())

            h5_id[counter] = text_uuid
            h5_image[counter] = best_image.reshape((1, - 1))  # np.asarray(cropped_image).reshape((batch_size, -1))
            h5_label[counter] = label
            h5_writer[counter] = str(params[WRITER])  # c_sample[1][counter].cpu()
            h5_original[counter] = params[ORIGINAL] .reshape((1, - 1)) # c_sample[2][counter]
            h5_style_sample[counter] = params[STYLE_SAMPLE].reshape((1, - 1))
            h5_name[counter] = str(name)
            counter = counter +1



def sampleselection_parse_args():
    parser = ArgumentParser()
    #TODO diffusion standard config

    parser.add_argument('--writer_tmp', type=str, default=os.path.join(os.getcwd(),"WriterCkpts"))
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--HTRConfig', type=str,default="HTR768x768.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="demo.yaml") #TODO is this the correct DL?
    parser.add_argument('--removeNewLine', type=str, default=False)
    parser.add_argument('--overWriteWriterCkpt', type=str, default=True)
    parser.add_argument('--sampled_h5file_path', type=str,default=r"TODO_SET_YOUR_PATH_TO_sampled_h5_file.h5")
    parser.add_argument('--savepath', type=str,default=os.path.join(os.getcwd(),"SampleSelectionDatasets"))

    return parser.parse_args()


if __name__ == "__main__":
    cfg = sampleselection_parse_args()

    os.makedirs(cfg.savepath,exist_ok=True)
    os.makedirs(cfg.writer_tmp,exist_ok=True)

    args = OmegaConf.load(get_yaml(r"Dataloaders",  cfg.DataloaderConfigFile))
    size = (args["params"]["size"][0],args["params"]["size"][1])
    z_shape = (args["params"]["z_shape"][0],args["params"]["z_shape"][1])

    img_dict, params_dict,names = get_datasets([cfg.sampled_h5file_path],size,cfg.removeNewLine)

    htr_model = instantiate_completely("HTR",cfg.HTRConfig)
    htr_model.cuda()
    htr_model.eval()

    writer_model = WriterSelect(cfg.DataloaderConfigFile,cfg.writer_tmp,overwrite=cfg.overWriteWriterCkpt,
                                application_reference_dl=r"Dataloaders",esvm=False,
                                ipca_comps=64,pca_comps=128,n_mvlad=5)

    get_full_ranking(img_dict, params_dict, names, htr_model, writer_model, z_shape, size, cfg.savepath,
                         cfg.name)
    #else:
    #    get_best_ranking(img_dict, params_dict, names, htr_model, writer_model, z_shape, size, save_path,
    #                     cfg.DiffusionConfig+"-"+cfg.name,
    #                     cer_importance,writer_importance)


