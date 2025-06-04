""" Modules for analysis of dimensionality of Nenya data."""
import os

from nenya.pca import fit_latents

def pca_latents(dataset:str):

    # Load
    key = 'valid'
    if dataset == 'MODIS_SST':
        filename = 'MODIS_R2019_2004_95clear_128x128_latents_std.h5'
        outfile='pca_latents_MODIS_SST.npz'
        path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2',
                        'Nenya', 'latents/MODIS_R2019_v4_REDO',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm')
    elif dataset == 'VIIRS_SST':
        filename = 'VIIRS_2013_98clear_192x192_latents_viirs_std_train.h5'
        outfile='pca_latents_VIIRS_SST.npz'
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
        mnist_path = os.getenv('OS_DATA')
        mnist_path = os.path.join(mnist_path, 'Natural', 'MNIST', 'Info')
        path = mnist_path
        filename = 'mnist_resampled_latents.h5'
        outfile='pca_latents_MNIST.npz'
        key = None
    else:
        raise IOError("Bad dataset: {}".format(dataset))

    fit_latents(os.path.join(path, filename), outfile, key=key)


# Command line execution
if __name__ == '__main__':
    # PCA MODIS SST
    #pca_latents('MODIS_SST')

    #  VIIRS SST
    #pca_latents('VIIRS_SST')

    #  LLC SST
    pca_latents('LLC_SST')

    # MNIST
    #pca_latents('MNIST')