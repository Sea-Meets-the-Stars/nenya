""" Module PCA analysis of the latent space """

import os
import h5py
import numpy as np
from sklearn import decomposition

def fit_latents(latents_file:str, 
                outfile:str,  
                key:str=None):
    """ Fit PCA to latents file
    Args:
        latents_file (str): Latents file
        outfile (str): Output file
        key (str):  If provided, restrict to this key in the latents file.
                if None, use the whole file.
    """
    # Load
    f = h5py.File(latents_file, 'r')
    keys = list(f.keys()) if key is None else [key]
    latents = []
    for key in keys:
        if key not in f:
            raise IOError(f"Key '{key}' not found in {latents_file}")
        print(f"Loading latents for key: {key}")
        # Load latents
        latents.append(f[key][:])
    latents = np.concatenate(latents, axis=0)
    f.close()

    # PCA
    print(f"Fitting PCA on {latents.shape[0]} samples with {latents.shape[1]} features")
    pca_fit = decomposition.PCA().fit(latents)

    # Save
    coeff = pca_fit.transform(latents)

    #
    outputs = dict(Y=coeff,
                M=pca_fit.components_,
                mean=pca_fit.mean_,
                explained_variance=pca_fit.explained_variance_ratio_)

    # Save
    print(f"Saving: {outfile}")
    np.savez(outfile, **outputs)