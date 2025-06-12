""" utilities for nenya analysis
"""
import os
from importlib import reload
import h5py

from nenya.train import main as train_main
from nenya import latents_extraction
from nenya import analysis
from nenya import plotting

from IPython import embed

def evaluate(opts_file:str):
    # Evaluate the model for MODIS native
    latents_extraction.evaluate(opts_file,
                os.path.join(modis_path, 'PreProc', 'train_MODIS_2021_128x128.h5'),
                local_model_path=modis_path,
                use_gpu=False,
                debug=False, clobber=True)

def chk_latents(dataset:str, latents_file:str, preproc_file:str,
                query_idx:int, partition:str='train', top_N:int=5):

    # Grab the latents
    with h5py.File(latents_file, 'r') as f:
        latents = f[partition][:]
        print(f"Latents shape: {latents.shape}")


    # Closest
    closest_idx, similarities = analysis.find_closest_latents(latents, query_idx)
    indices = [query_idx]+closest_idx[:top_N].tolist()

    # Grab the images
    with h5py.File(preproc_file, 'r') as f:
        images = [f[partition][idx] for idx in [query_idx]+closest_idx[:top_N].tolist()]
        print(f"Grabbed {len(images)} images for plotting including the query.")

    # Plot
    #embed(header='53 of nenya')
    #reload(plotting)
    plotting.closest_latents(images, indices, similarities,
                          output_png=f'nenya_{dataset}_{partition}_chk_latents_{query_idx}.png')


def train(opts_file:str, load_epoch:int=None, debug:bool=False):
    # Train the model
    train_main(opts_file, debug=debug, load_epoch=load_epoch)


def return_task():
    import sys

    if len(sys.argv) == 1:
        print("Usage: python nenya_ImageNet.py <task>")
        print("Tasks: train, evaluate, chk_latents")
    elif len(sys.argv) > 2:
        print("Too many arguments. Only one task is allowed.")
        sys.exit(1)
    elif len(sys.argv) == 2:
        task = sys.argv[1].lower()
        if task not in ['train', 'evaluate', 'chk_latents']:
            print(f"Unknown task: {task}. Use 'train', 'evaluate', or 'chk_latents'.")
            sys.exit(1)
        print(f"Running task: {task}")
        task = sys.argv[1].lower()

    return task