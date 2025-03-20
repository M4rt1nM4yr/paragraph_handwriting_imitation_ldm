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




alphabet = Alphabet(dataset="all")


def get_combined_score(htr_rankings, writer_rankings,cer_importance=1.0001,writer_importance=1.0,rank=0):

    max_score = len(htr_rankings)
    ranking = np.zeros((max_score))

    current_score = max_score
    for i in range(len(htr_rankings)):
        ranking[htr_rankings[i]] = ranking[htr_rankings[i]] + current_score *cer_importance
        ranking[writer_rankings[i]] = ranking[writer_rankings[i]] + current_score *writer_importance
        current_score = current_score -1

    return ranking.argsort()[::-1][rank]



"""
    transforms the datasets into a dictionary
    img_dict:  name -> images
    params_dict: name -> parameters
    names: list of all sample names
"""

def get_datasets(file_paths,size,remove_newline,batch_size=10):

    img_dict = dict()
    params_dict = dict()
    names = []

    for file in file_paths:
        dataset = h5py.File(file, "r")
        length = len(dataset["writer"])


        for i in range(length):
            batch_size = batch_size
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
                    "name": name,
                    "writer_score": dataset["writer_score"][i],
                    "cer_score": dataset["cer_score"][i],
                }
                names.append(name)
    return img_dict, params_dict,names



def get_best_ranking(img_dict, params_dict,names,cer_importance,writer_importance,save_path,max_rank=1):
    os.makedirs(save_path,exist_ok=True)
    for rank in range(max_rank):

        with h5py.File(os.path.join(save_path, "rank_"+str(rank+1)+"_x.h5"), "w") as synthetic_data:
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

            stds = []
            avg_cer = 0.0

            counter = 0
            for name in tqdm(names):
                params = params_dict[name]
                images = img_dict[name]

                htr_cers = params_dict[name]["cer_score"]
                writer_scores = params_dict[name]["writer_score"]

                cer_ranking = np.array(htr_cers).argsort()
                writer_ranking = np.array(writer_scores).argsort()
             #   print(htr_cers)
                best_idx = get_combined_score(cer_ranking,writer_ranking,cer_importance,writer_importance,rank)
                best_image = images[best_idx]
                avg_cer = avg_cer + htr_cers[best_idx]

                stds.append(np.array(htr_cers).std())

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


def buildranking_parse_args():
    parser = ArgumentParser()
    #TODO diffusion standard config

    parser.add_argument('--ranking_h5file_path', type=str,default=r"TODO insert your ranking file here")
    parser.add_argument('--save_path', type=str,default=os.path.join(os.getcwd(),"SampleSelectionDatasets"))
    parser.add_argument('--maxrank', type=int,default=10)
    parser.add_argument('--cerImportance', type=float,default=1.0)
    parser.add_argument('--writerImportance', type=float,default=1.0)
    parser.add_argument('--removeNewLine', type=str, default=False)
    parser.add_argument('--name', type=str, default="--default")

    return parser.parse_args()

if __name__ == "__main__":

    cfg = buildranking_parse_args()
    size = (768,768)

    note = cfg.name+"-"+"CER_"+str(cfg.cerImportance)+"_W_"+str(cfg.writerImportance)+""
    os.makedirs(cfg.save_path,exist_ok=True)

    img_dict, params_dict,names = get_datasets([cfg.ranking_h5file_path],size,cfg.removeNewLine)

    get_best_ranking(img_dict, params_dict,names,cfg.cerImportance,cfg.writerImportance,
                     cfg.save_path,max_rank=cfg.maxrank)


