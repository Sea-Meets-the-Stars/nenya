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

def evaluate(opts_file:str, preproc_file:str, latents_file:str=None,
             local_model_path:str=None, use_gpu:bool=False, clobber:bool=False):
    """
    Evaluate the latents extraction process using the specified options and preprocessing files.

    Args:
        opts_file (str): Path to the options file containing configuration settings.
        preproc_file (str): Path to the preprocessing file required for evaluation.
        local_model_path (str, optional): Path to the local model file. Defaults to None.
        latents_file (str, optional): Path to the file where latents will be stored. Defaults to None.
        clobber (bool, optional): If True, overwrite existing latents file. Defaults to False.
        use_gpu (bool, optional): Flag indicating whether to use GPU for evaluation. Defaults to False.

    Returns:
        None: This function does not return a value. It performs evaluation and may modify files or output logs.
    """
    latents_extraction.evaluate(opts_file,
                preproc_file,
                local_model_path=local_model_path,
                latents_file=latents_file,
                use_gpu=use_gpu,
                debug=False, clobber=clobber)

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