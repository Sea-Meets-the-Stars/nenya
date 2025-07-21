""" Modules for analysis of dimensionality of Nenya data."""
import os

from nenya.pca import fit_latents
import info_defs

def pca_latents(dataset:str):

    # Load
    Nmax = None
    key = 'valid'
    pdict = info_defs.grab_paths(dataset)
    if dataset == 'MODIS_SST_2km':
        key = None
    elif dataset == 'MODIS_SST_2km_sub':
        Nmax = 200000
    elif dataset == 'MODIS_SST':
        key = None
    elif dataset == 'VIIRS_SST':
        key = None
    elif dataset == 'VIIRS_SST_2km':
        key = None
    elif 'LLC_SST' in dataset:
        key = None
    elif dataset == 'MNIST':
        key = None
    elif dataset == 'SWOT_L3':
        key = None
    elif dataset == 'ImageNet':
        key = None
    else:
        raise IOError("Bad dataset: {}".format(dataset))

    fit_latents(pdict['latents_file'], pdict['pca_file'], key=key, Nmax=Nmax)


# Command line execution
if __name__ == '__main__':
    # PCA MODIS SST
    #pca_latents('MODIS_SST_2km')
    #pca_latents('MODIS_SST_2km_sub')
    #pca_latents('MODIS_SST')

    #  VIIRS SST
    pca_latents('VIIRS_SST')
    #pca_latents('VIIRS_SST_2km')

    #  LLC SST
    #pca_latents('LLC_SST_nonoise')
    #pca_latents('LLC_SST_noise')

    # MNIST
    #pca_latents('MNIST')

    # ImageNet
    #pca_latents('ImageNet')

    # SWOT L3
    #pca_latents('SWOT_L3')