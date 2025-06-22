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
    elif dataset == 'VIIRS_SST_2km':
        filename = 'VIIRS_2013_98clear_192x192_latents_viirs_std_train.h5'
        outfile='pca_latents_VIIRS_SST_2km.npz'
        path = os.path.join(os.getenv('OS_SST'), 'VIIRS',
                        'Nenya', 'latents', 'VIIRS_v1') 
    elif dataset == 'LLC_SST':
        filename = 'LLC_nenya_training.h5'
        outfile='pca_latents_LLC_SST.npz'
        path = os.path.join(os.getenv('OS_OGCM'), 'LLC',
                        'Nenya', 'latents', 'LLC_v1',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm') 
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
    pca_latents('MODIS_SST')

    #  VIIRS SST
    #pca_latents('VIIRS_SST')

    #  LLC SST
    #pca_latents('LLC_SST')

    # MNIST
    #pca_latents('MNIST')

    # ImageNet
    #pca_latents('ImageNet')

    # SWOT L3
    #pca_latents('SWOT_L3')