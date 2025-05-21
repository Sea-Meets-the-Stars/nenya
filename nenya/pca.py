""" Module PCA analysis of the latent space """

import os
import h5py
import numpy as np
from sklearn import decomposition

def fit_latents(latents_file:str, 
                outfile:str,  
                key:str='valid'):
    """ Fit PCA to latents file
    Args:
        latents_file (str): Latents file
        outfile (str): Output file
        key (str): Key in latents file. Default is 'valid'
    """
    # Load
    f = h5py.File(latents_file, 'r')
    latents = f[key][:]
    f.close()

    # PCA
    print("Fitting PCA")
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