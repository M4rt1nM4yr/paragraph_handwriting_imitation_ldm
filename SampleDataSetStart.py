from src.SamplingModules.ConditionalDataSetSampler import start_sampling
import argparse, os, sys, glob, datetime, yaml
import Parameters as pa
from argparse import ArgumentParser


def diffusion_dataset_sample_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--DiffusionConfig', type=str, default="ours.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="demo.yaml")  # TODO is this the correct DL?
    parser.add_argument('--repeats', type=int, default=10) # How often do you want to sample the same sample
    parser.add_argument('--name', type=str, default="--LatentDiffusion_Demo_DatasetSampling")
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidanceScale', type=float, default=2.5)
    parser.add_argument('--logdir', type=str,default=os.path.join(os.getcwd(),"SampledDataset"))

    return parser.parse_args()

if __name__ == "__main__":

    cfg = diffusion_dataset_sample_parse_args()

    sampling_hyper_parameters = (cfg.steps,cfg.guidanceScale,0.0,0.0)
    start_sampling(cfg.DiffusionConfig,cfg.repeats, logdir=cfg.logdir,
                   sampling_hyper_parameters=sampling_hyper_parameters,description=cfg.name,
                   dl_config = cfg.DataloaderConfigFile,dl_apply="Dataloaders")