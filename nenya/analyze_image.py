""" Module to explore an input image """
import os
import numpy as np
from importlib import reload

import umap

import torch

from nenya import latents_extraction
from nenya import io as nenya_io
from nenya import nenya_umap
from nenya import params

from IPython import embed

def get_latents(img:np.ndarray, 
                model_file:str, 
                opt:params.Params):
    """ Get the Nenya latents for an input image

    Args:
        img (np.ndarray): 
            Input image (64,64) or (1,64,64)
        model_file (str): 
            Full path to the Nenya model file, e.g. $OS_SST/MODIS_L2/Nenya/models/v4_last.pth
            Must be on the local filesystem (s3 is not allowed)
        opt (ulmo_io.Params): 
            Parameters for the Nenya model, e.g. 
                ~/runs/viiirs/opts_viirs_v1.json

    Returns:
        tuple: latents (np.ndarray), pre-processed image (np.ndarray) 
    """

    # Reshape image as need be
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0) 
        img = np.expand_dims(img, axis=0) 
        img = np.repeat(img, 3, axis=1)

    # Pre-process
    pp_img = img - np.mean(img)
    
    # DataLoader
    dset = torch.utils.data.TensorDataset(torch.from_numpy(pp_img).float())
    data_loader = torch.utils.data.DataLoader(
        dset, batch_size=1, shuffle=False, collate_fn=None,
        drop_last=False, num_workers=1)

    # Time to run
    latent_dict = latents_extraction.model_latents_extract(
        opt, None, model_file, partitions=['valid'],
         in_loader=data_loader)

    # Return
    latents = latent_dict['valid']
    return latents, pp_img

def calc_DT(images, random_jitter:list,
              verbose=False):
    """Calculate DT for a given image or set of images
    using the random_jitter parameters

    Args:
        images (np.ndarray): 
            Analyzed shape is (N, 64, 64)
            but a variety of shapes is allowed
        random_jitter (list):
            range to crop, amount to randomly jitter
        verbose (bool, optional): Print out progress. Defaults to False.
    Returns:
        np.ndarray or float: DT
    """
    if verbose:
        print("Calculating T90")

    # If single image, reshape into fields
    single = False
    if len(images.shape) == 4:
        fields = images[...,0,...]
    elif len(images.shape) == 2:
        fields = np.expand_dims(images, axis=0) 
        single = True
    elif len(images.shape) == 3:
        fields = images
    else:
        raise IOError("Bad shape for images")

    # Center
    xcen = fields.shape[-2]//2    
    ycen = fields.shape[-1]//2    
    dx = random_jitter[0]//2
    dy = random_jitter[0]//2
    if verbose:
        print(xcen, ycen, dx, dy)
    
    # Calculate T90, T10
    if verbose:
        print("Calculating T90")
    T_90 = np.percentile(fields[..., xcen-dx:xcen+dx,
        ycen-dy:ycen+dy], 90., axis=(1,2))
    if verbose:
        print("Calculating T10")
    T_10 = np.percentile(fields[..., xcen-dx:xcen+dx,
        ycen-dy:ycen+dy], 10., axis=(1,2))
    #T_10 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
    #    10., axis=(1,2))
    DT = T_90 - T_10

    # Return
    if single:
        return DT[0]
    else:
        return DT

def umap_image(nenya_model:str, img:np.ndarray):
    """UMAP the input image

    Args:
        nenya_model (str): Nenya model name, e.g. v4, v5
        img (np.ndarray): Input image

    Returns:
        tuple: UMAP embedding, pre-processed image (np.ndarray), 
            name of table file associated with this UMAP (str), DT (float)
            latents (np.ndarray)
    """

    # Load opt
    opt, nenya_model_file = nenya_io.load_opt(nenya_model)

    # Calculate latents
    latents, pp_img = get_latents(img, nenya_model_file, opt)

    # DT40
    DT = calc_DT(img, opt.random_jitter)
    print("Image has DT={:g}".format(DT))

    # UMAP me
    #embed(header='umap_image: about to embed')
    print("Embedding")
    reload(nenya_umap)
    latents_mapping, table_file = nenya_umap.load(nenya_model, DT=DT)
    embedding = latents_mapping.transform(latents)

    print(f'U0,U1 for the input image = {embedding[0,0]:.3f}, {embedding[0,1]:.3f}')

    # Return
    return embedding, pp_img, table_file, DT, latents