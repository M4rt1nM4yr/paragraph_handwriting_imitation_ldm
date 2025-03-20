# Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models
Our implementation is build upon the implementation of latent diffusion models from https://github.com/CompVis/latent-diffusion.

## Setup 
Hardware details: We recommend more than 8GB VRAM to run the latent diffusion model and a graphic card that supports Cuda.
The requirements are defined in enviornment.yaml. To train the models, we recommend around 40GB+ VRAM.

## How to Train the Model
1. Please set up your environment with the provided environment.yaml and download your desired datasets (we used IAM, and our synthetical data as IAM alone is probably not enough). We also recommend pretraining every model (besides the ldm) on line level before switching to paragraphs. (Finetuning on real data alone is also highly recommended).
2. Train the HTR-Model with the config file from configs/HTR/HTR768x768.yaml. (HTRTrainer.py)
3. Train the WriterCNN with the config file from configs/Writer/768x768WriterCNN.yaml (WriterSequenceTrainer.py) -- Note if you train on a different dataset you might have to adjust the number of writers to your dataset
4. Train the KL-regularized Autoencoder with the config file from configs->AutoKL->768x768AutoKL.yaml -- Attention remember to set the checkpoints for the HTR and WRITER CNN in the config file! (AutoEncoderTrainer.py)
5. Train the Latent Diffusion Model with the config file from configs->Diffusion->ldm->ours -- Attention remember to set the checkpoints for the WriterCNN and KL-regularized Autoencoder here. Furthermore, you have to use the special diffusion dataloader configs because of the augmentations. (Diffusiontrainer.py)
6. Once you have trained the latent diffusion model, you can use the demo.py to sample images with custom text with style samples provided in style samples. The style samples are created from the IAM database. Feel free to include your style samples ( we require them to have a 768x768 resolution). Note that we have added more sample steps than necessary as a default setting to showcase the diffusion process as an additional output.

If you want to go further and sample the entire dataset to test our ranking method, you will need a few more additional steps:
7. To sample an entire dataset, please use SampleDataSetStart.py and provide the necessary specifications. Please set repeat to a value higher than one if you want to use ranking.
8. Use SampleSelection.py to assign a style and text evaluation for every sampled image. 
9. Afterward, use BuildFromRankings.py to build the entire ranking.


## Using the Model with Pre-trained Checkpoints
To use the model with the pre-trained checkpoints, follow these steps:

1. **Download the Checkpoint**
   - Download the pre-trained checkpoint from [Google Drive](https://drive.google.com/diffusion_checkpoint.ckpt).

2. **Setup the Environment**
   - Ensure your environment is set up as per the `environment.yaml` file.

3. **Load the Checkpoint**
   - Use the provided scripts to load the checkpoint into your model. You may need to specify the path to the checkpoint file in the configuration or script.

4. **Run the Model**
   - Execute the `demo.py` script to generate samples using the pre-trained model. Ensure that the path to the checkpoint is correctly set in the script or configuration.

This will allow you to utilize the pre-trained model for generating handwriting samples with the desired styles.

## Datasets
- [https://fki.tic.heia-fr.ch/databases/iam-handwriting-database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (primarily)
- Synthetic data generated with fonts. (We plan to make our synthetical training data publicly available at a later date.)


## Citation
If you use this work, please cite:

```bibtex
@misc{mayr2024zeroshotparagraphlevelhandwritingimitation,
      title={Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models}, 
      author={Martin Mayr and Marcel Dreier and Florian Kordon and Mathias Seuret and Jochen Zöllner and Fei Wu and Andreas Maier and Vincent Christlein},
      year={2024},
      eprint={2409.00786},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00786}, 
}
```
