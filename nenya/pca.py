""" Module PCA analysis of the latent space """

import os
import h5py
import numpy as np
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

from IPython import embed

def fit_latents(latents_file:str, 
                outfile:str,  
                Nmax:int=None,
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

    # Chop down?
    if Nmax is not None:
        if Nmax > latents.shape[0]:
            raise ValueError(f"Nmax ({Nmax}) is larger than the number of samples ({latents.shape[0]})")
        print(f"Chopping down to {Nmax} samples")
        latents = latents[:Nmax, ...]

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

def generate_eigenmode_with_regularization(model, target_latent:np.ndarray, 
                                           image_shape:np.ndarray, 
                                           num_iterations=1000, lr:float=0.01, 
                                           clamp_value:float=None,
                                           tv_weight:float=1e-4, l2_weight:float=1e-6):
    """
    Enhanced version with total variation and L2 regularization for smoother images.
    """
    generated_image = torch.randn(1, *image_shape, requires_grad=True, device=target_latent.device)
    optimizer = torch.optim.Adam([generated_image], lr=lr)

    # Convert target_latent to tensor
    target_latent = torch.tensor(target_latent, dtype=torch.float32, device=target_latent.device)

    if clamp_value is None:
        clamp_value = 5.0
    
    model.eval()
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Main reconstruction loss (consider cosine similarity instead)
        predicted_latent = model(generated_image)
        reconstruction_loss = F.mse_loss(predicted_latent, target_latent.unsqueeze(0))
        
        # Total variation regularization (encourages smoothness)
        tv_loss = torch.sum(torch.abs(generated_image[:, :, :, :-1] - generated_image[:, :, :, 1:])) + \
                  torch.sum(torch.abs(generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]))
        
        # L2 regularization (prevents extreme pixel values)
        l2_loss = torch.sum(generated_image ** 2)
        
        # Combined loss
        total_loss = reconstruction_loss + tv_weight * tv_loss + l2_weight * l2_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Clamp pixel values to reasonable range
        with torch.no_grad():
            generated_image.clamp_(-1*clamp_value, 1*clamp_value)
        
        if i % 100 == 0:
            print(f"Iter {i}: Recon={reconstruction_loss.item():.6f}, "
                  f"TV={tv_weight*tv_loss.item():.6f}, L2={l2_weight*l2_loss.item():.6f}")

            # Check cosine similarity of the generated latent with the target latent
            with torch.no_grad():
                predicted_latent = model(generated_image)
                cosine_sim = F.cosine_similarity(predicted_latent, target_latent.unsqueeze(0))
                print(f"Cosine Similarity: {cosine_sim.item():.6f}")

    # Once more, just in case
    with torch.no_grad():
        predicted_latent = model(generated_image)
        cosine_sim = F.cosine_similarity(predicted_latent, target_latent.unsqueeze(0))
        print(f"Final cosine Similarity: {cosine_sim.item():.6f}")
    
    # Convert to numpy and detach
    generated_image = generated_image.detach()
    generated_image = generated_image.cpu().numpy()
    generated_image = generated_image.squeeze(0)  # Remove batch dimension
    #
    return generated_image, cosine_sim.detach().cpu().numpy()[0]