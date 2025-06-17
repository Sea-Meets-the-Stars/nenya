""" utilities for nenya analysis
"""
import os
from importlib import reload
import numpy as np
import h5py


from wrangler.plotting import cutout

from nenya.train import main as train_main
from nenya import latents_extraction
from nenya import analysis
from nenya import pca
from nenya import plotting
from nenya import params 
from nenya import io as nenya_io

from IPython import embed

def evaluate(opts_file:str, preproc_file:str, latents_file:str=None,
             local_model_path:str=None, use_gpu:bool=False, clobber:bool=False,
             base_model_name:str='last.pth', debug:bool=False):
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
                debug=debug, clobber=clobber,
                base_model_name=base_model_name)

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
    plotting.closest_latents(images, indices, similarities,
                          output_png=f'nenya_{dataset}_{partition}_chk_latents_{query_idx}.png')


def train(opts_file:str, load_epoch:int=None, debug:bool=False):
    """
    Train the model using the specified options file.

    Args:
        opts_file (str): Path to the options file containing training configurations.
        load_epoch (int, optional): Epoch number to load for resuming training. Defaults to None.
        debug (bool, optional): Flag to enable debug mode. Defaults to False.

    Returns:
        None
    """
    # Train the model
    train_main(opts_file, debug=debug, load_epoch=load_epoch)

def find_eigenmodes(opt_path:str, pca_file:str, image_shape:tuple, output_file:str, 
                    Neigenmodes:int=10, use_gpu:bool=False, clamp_value:float=None, 
                    local_model_path:str=None, base_model_name:str='last.pth',
                    num_iterations:int=1000,
                    tv_weight:float=0.0, show:bool=False,
                    debug:bool=False):
    """
    Find and visualize the specified eigenmode of a model.

    Args:
        model_file (str): Path to the model file.
        pca_file (str): Path to the PCA file containing eigenmodes.
        eigenmode (int): Index of the eigenmode to visualize. Defaults to 0.
        output_file (str): Path to save the output visualization.
        n_samples (int, optional): Number of samples to use for visualization. Defaults to 1000.
        debug (bool, optional): Flag to enable debug mode. Defaults to False.

    Returns:
        None
    """
    # Load model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # Model name and opt
    opt, model_name = nenya_io.load_model_name(
        opt_path, local_model_path=local_model_path,
        base_model_name=base_model_name)

    # Load model
    model = nenya_io.load_model(model_name, opt, use_gpu,
                               remove_module=True, 
                               weights_only=False)

    # Load the PCA model
    d = np.load(pca_file)

    
    # Run it
    eigen_images = []
    similarities = []
    for ss in range(Neigenmodes):
        # Grab the eigenmode
        eigenmode = d['M'][ss, :]
        # Generate the eigenmode with regularization
        img, cosi = pca.generate_eigenmode_with_regularization(model, eigenmode, image_shape, 
            tv_weight=tv_weight, clamp_value=clamp_value, num_iterations=num_iterations)
        eigen_images.append(img)
        similarities.append(cosi)
        # Show the image?
        if show:
            ax = cutout.show_image(img[0], show=True)
            ax.set_title=f'Eigenmode {ss+1}'
        if debug: 
            embed(header=f"Generated eigenmode {ss+1}/{Neigenmodes} with shape {img.shape}")

    # Save the eigenmodes
    if not debug:
        print(f"Saving eigenmodes to: {output_file}")
        np.savez(output_file, eigen_images=np.array(eigen_images), 
            similarities=np.array(similarities),
            eigenmodes=d['M'][:Neigenmodes, :])


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
        if task not in ['train', 'evaluate', 'chk_latents', 'eigenimages']:
            print(f"Unknown task: {task}. Use 'train', 'evaluate', 'eigenimages', or 'chk_latents'.")
            sys.exit(1)
        print(f"Running task: {task}")
        task = sys.argv[1].lower()

    return task