from src.SamplingModules.ConditionalSampler import start_sampling
import argparse, os, sys, glob, datetime, yaml
import Parameters as pa
from argparse import ArgumentParser


def short_sampling_parse_args():
    parser = ArgumentParser()
    #TODO diffusion standard config
    parser.add_argument('--DiffusionConfig', type=str,default="ours.yaml")
    parser.add_argument('--DataloaderConfigFile', type=str, default="demo.yaml") #TODO is this the correct DL?
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--Sampled_demo_images',type=str,default=os.path.join(os.getcwd(),"demo_images"))
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--useStylExamples', type=str, default=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--guidanceScale', type=float, default=2.5)

    return parser.parse_args()



"""
    Make sure the style sample has a resolution of 768x768!!!
"""


if __name__ == "__main__":

    cfg = short_sampling_parse_args()

    conditioning_string = "Hello ladies, gentlemen and others,\n" \
                          "I would like to tell you a story about\n" \
                          "the quick brown fox that jumps over the\n" \
                          "lazy dog. He is in fact very cute and\n" \
                          "well behaved. I promise. But you should\n" \
                          "not annoy him! Otherwise, he will retaliate\n" \
                          "like a wild dog or animal. So he could\n" \
                          "bite you or not. It is his choice really.\n" \

    #set this to true for the noNL model
    remove_newline = False

    start_sampling(batch_size=cfg.batch_size,diffusion_config=cfg.DiffusionConfig,conditioning_string=conditioning_string,
                   logdir=cfg.Sampled_demo_images, set="test",remove_newline=remove_newline,steps=cfg.steps,
                   guidanceScale = cfg.guidanceScale,dl_config = cfg.DataloaderConfigFile,
                   dl_apply="Dataloaders",from_example=cfg.useStylExamples)