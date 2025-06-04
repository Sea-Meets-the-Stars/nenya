""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np
import h5py

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from remote_sensing.plotting import utils as rsp_utils
from wrangler import s3_io

from nenya import plotting as nenya_plotting

mpl.rcParams['font.family'] = 'stixgeneral'

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import reconstruct

from IPython import embed


def fig_pca(outfile:str='fig_pca_variance.png',
            exponent:float=-0.5): 

    # Load PCAs
    ds = []
    datasets = ['MODIS_SST', 'VIIRS_SST', 
                'LLC_SST', 
                #'SWOT_SSR', 
                'MNIST']
    #datasets = ['MODIS_SST']
    for dataset in datasets:
        d = np.load(f'../Analysis/pca_latents_{dataset}.npz')
        ds.append(d)

    # 
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])
    for ss, d in enumerate(ds):
        ax.plot(np.arange(d['explained_variance'].size)+1, 
                d['explained_variance'], 'o', label=datasets[ss])
        if ss == 0:
            xs = np.arange(d['explained_variance'].size)+1

    ys = d['explained_variance'][10] * (xs/xs[10])**(exponent) 
    ax.plot(xs, ys, '--', color='gray', label=f'Power law: {exponent}')
    # Label
    ax.set_ylabel('Variance explained per mode')
    ax.set_xlabel('Number of PCA components')
    #
    #ax.set_xlim(0,10.)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Minor ticks
    ax.minorticks_on()
    # Horizontal line at 0
    #ax.axhline(0., color='k', ls='--')

    #loc = 'upper right' if ss == 1 else 'upper left'
    ax.legend(fontsize=15, loc='lower left')

    # Turn on grid
    ax.grid(True, which='both', ls='--', lw=0.5)

    rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def dataset_path(dataset:str):
    if dataset == 'SWOT':
        path = os.path.join(os.getenv('SWOT_PNGs'),
                        'models', 'SWOT',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm')
    elif dataset == 'VIIRS':
        path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info',
                        'models', 'VIIRS_N21',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm')
    elif dataset == 'MODIS':
        path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info',
                        'models', 'MODIS_2021',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm')
    elif dataset == 'MNIST':
        path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info',
                        'models', 'MNIST',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm')
    else:
        raise ValueError(f"Dataset {dataset} not supported for learning curve plotting.")
    return path

def fig_learning_curves(outfile:str=f'fig_learning_curves.png'):
    """Plot the learning curves for SWOT, VIIRS, MODIS and MNIST datasets."""
    
    # Define the datasets
    datasets = ['VIIRS', 'MODIS', 'MNIST']
    
    # Create a figure
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    for ss, dataset in enumerate(datasets):
        path = dataset_path(dataset)
        valid_file = os.path.join(path, 'learning_curve',
                                  f'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_valid.h5')
        train_file = os.path.join(path, 'learning_curve',
                                  f'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_train.h5')
        with s3_io.open(valid_file, 'rb') as f:
            valid_hf = h5py.File(f, 'r')
            loss_valid = valid_hf['loss_valid'][:]
        with s3_io.open(train_file, 'rb') as f:
            train_hf = h5py.File(f, 'r')
            loss_train = train_hf['loss_train'][:]

        # Plot
        if ss == 0:
            lbl0 = f'{dataset} validation'
            lbl1 = f'{dataset} training'
        else:
            lbl0 = f'{dataset} validation'
            lbl1 = f'{dataset} training'
        ax.plot(np.arange(loss_valid.size)+1, loss_valid, label=lbl0, lw=3)
        ax.plot(np.arange(loss_train.size)+1, loss_train, label=lbl1, lw=3)

        
        nenya_plotting.learn_curve(valid_file, train_file, 
                                   ax=ax, ylog=True)
    
    axs[-1].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss (log scale)')
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_learning_curve(dataset:str='SWOT'):
    outfile=f'fig_{dataset}_learning_curve.png'
    # Load the learning curve files
    path = dataset_path(dataset)
    valid_file = os.path.join(path, 'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_valid.h5')
    train_file = os.path.join(path, 'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_train.h5')
    # Plot the learning curve
    nenya_plotting.learn_curve(valid_file, train_file, 
                                   outfile=outfile, ylog=True)

def fig_swot_umap(outfile:str='fig_swot_umap_gallery.png'):
    swot_path = os.getenv('SWOT_PNGs')
    tbl_file = os.path.join(swot_path,'Pass_006.parquet')
    img_file = os.path.join(swot_path,'Pass_006.h5')
    nenya_plotting.umap_gallery(tbl_file, img_file, 
                                dxv=1.0, dyv=1.0,
                                #scl_inset=(0.9,0.9),
                                in_vmnx=(-0.5, 1.5),
                                cbar_lbl='SWOT/SSR',
                                outfile=outfile,
                                debug=True,
                                cmap="Greys",)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # PCA variaince
    if flg == 1:
        fig_pca()

    # SWOT learning curve
    if flg == 30:
        #fig_learning_curve('SWOT')
        #fig_learning_curve('VIIRS')
        #fig_learning_curve('MODIS')
        fig_learning_curve('MNIST')

    # SWOT UMAP gallery
    if flg == 40:
        fig_swot_umap()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)