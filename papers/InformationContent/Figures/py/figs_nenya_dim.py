""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np
import h5py

import torch
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from remote_sensing.plotting import utils as rsp_utils

from wrangler import s3_io

from nenya import plotting as nenya_plotting
from nenya import params 
from nenya import io as nenya_io
from nenya import analysis

mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

from IPython import embed

# Color eict
cdict = {}
cdict['MODIS'] = '#1f77b4'  # Blue
cdict['VIIRS'] = '#ff7f0e'  # Orange
cdict['LLC'] = '#2ca02c'  # Green
# Gray
cdict['MNIST'] = '#7f7f7f'  # Gray
# Red
cdict['SWOT_L3'] = '#d62728'  # Red
# Black
cdict['ImageNet'] = '#000000'  # Black

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

def fig_pca(outfile:str='fig_pca_variance.png',
            datasets:list=None, cumulative:bool=False,
            frac_remain:bool=False,
            show_cum_point:float=None,
            exponent:float=-0.5): 
    # Cumulative?
    if cumulative:
        if 'variance' in outfile:
            outfile = outfile.replace('variance', 'cumulative')

    # Load PCAs
    if datasets is None:
        datasets = ['MODIS_SST', 'MODIS_SST_2km', #'MODIS_SST_2km_sub',
                #'VIIRS_SST_2km', 
                'LLC_SST_nonoise', 
                'SWOT_L3', 
                #'SWOT_SSR', 
                'MNIST',
                'ImageNet',
                ]
    #datasets = ['MODIS_SST']
    clrs = []
    ds = []
    for dataset in datasets:
        if 'SST' in dataset:
            clr = cdict[dataset.split('_')[0]]
        else:
            clr = cdict[dataset]
        
        pca_file = f'../Analysis/pca_latents_{dataset}.npz'
        print(f"Loading PCA file: {pca_file}")
        d = np.load(pca_file)
        ds.append(d)
        #
        clrs.append(clr)

    #embed(header='PCA Variance Explained 89')

    # 
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])
    for ss, d in enumerate(ds):
        if 'sub' in datasets[ss]:
            ls = '--' 
        elif '_noise' in datasets[ss]:
            ls = '--' 
        elif '2km' in datasets[ss]:
            ls = ':' 
        else:
            ls = '-'
        # Cumulative?
        cumsum = 1-np.cumsum(d['explained_variance'])
        if cumulative:
            yvals = cumsum
        elif frac_remain:
            cumsum = 1-np.cumsum(d['explained_variance'])
            yvals = d['explained_variance'] / cumsum
        else:
            yvals = d['explained_variance']
        ax.plot(np.arange(d['explained_variance'].size)+1, 
                yvals,  label=datasets[ss].replace('_','/'),
                color=clrs[ss], ls=ls)
        # Add cum point
        if show_cum_point is not None:
            imin = np.argmin(np.abs((1-cumsum) - show_cum_point))
            ax.plot(imin+1, yvals[imin], 'x', color=clrs[ss])

            
        if ss == 0:
            xs = np.arange(d['explained_variance'].size)+1

    ys = d['explained_variance'][10] * (xs/xs[10])**(exponent) 
    ax.plot(xs, ys, '--', color='gray', label=f'Power law: {exponent}')
    # Label
    if cumulative:
        ax.set_ylabel('Cumulative Variance explained per mode')
    else:
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



def fig_learning_curves(outfile:str=f'fig_learning_curves.png'):
    """Plot the learning curves for SWOT, VIIRS, MODIS and MNIST datasets."""
    
    # Define the datasets
    datasets = ['VIIRS', 'MODIS', 'MNIST', 'SWOT_L3']
    
    # Create a figure
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    for ss, dataset in enumerate(datasets):
        print(f'Processing dataset: {dataset}')
        clr = cdict[dataset]
        #path = dataset_path(dataset)
        pdict = info_defs.grab_paths(dataset)
        opt = params.Params('../Analysis/'+pdict['opts_file'])
        params.option_preprocess(opt)
        #embed(header=f"Learning curves for {dataset}")
        losses_train, losses_valid = nenya_io.losses_filenames(opt)

        valid_file = os.path.join(pdict['path'], losses_valid)
        train_file = os.path.join(pdict['path'], losses_train)
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
            lbl0 = None
            lbl1 = f'{dataset}'
        ax.plot(np.arange(loss_valid.size)+1, loss_valid, label=lbl0, lw=3, color=clr, ls='--')
        ax.plot(np.arange(loss_train.size)+1, loss_train, label=lbl1, lw=3, color=clr)

        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')

    ax.legend(fontsize=15, loc='upper right')

    rsp_utils.set_fontsize(ax, 21.)
    
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

def fig_eigenimages(dataset:str, cmap:str, Nimages:int=9, 
                    outroot:str='fig_eigenmodes'):

    outfile = f'{outroot}_{dataset}.png'

    # Load eigenimages
    d = np.load(f'../Analysis/{dataset}_eigenimages.npz')


    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(3,3)

    for ss in range(Nimages):
        img = d['eigen_images'][ss][0,...]

        ax = plt.subplot(gs[ss]) 
        _ = sns.heatmap(np.flipud(img), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1], 
                     ax=ax,
                     yticklabels=[], cmap=cmap, cbar=False) 
                     #cbar_kws={'label': clbl})# 'fontsize': 20})
        # Title
        title = f'Eigenmode {ss+1} sim={d["similarities"][ss]:.2f}'
        ax.set_title(title, fontsize=12)

    #rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_eigenmatches(dataset:str, cmap:str, Nimages:int=9, 
                     partition:str='train',
                    outroot:str='fig_eigenmatches'):

    outfile = f'{outroot}_{dataset}.png'

    # Load the PCA model
    pdict = info_defs.grab_paths(dataset)
    d = np.load(f'../Analysis/{pdict['pca_file']}')

    # Grab the latents
    with h5py.File(pdict['latents_file'], 'r') as f:
        latents = f[partition][:]

    # Open the preproc file
    preproc_file = pdict['preproc_file']
    preproc = h5py.File(preproc_file, 'r')

    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(3,3)

    for ss in range(Nimages):
        eigenmode = d['M'][ss, :]
        # Closest
        query_vector = eigenmode.reshape(1, -1)
        similarities = cosine_similarity(query_vector, latents)[0]
        # Sort
        sorted_indices = np.argsort(-similarities)
        similarities = similarities[sorted_indices]

        # Grab the image
        img = preproc[partition][sorted_indices[0]]
        if img.ndim == 3:
            img = img[0,...]

        ax = plt.subplot(gs[ss]) 
        _ = sns.heatmap(np.flipud(img), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1], 
                     ax=ax,
                     yticklabels=[], cmap=cmap, cbar=False) 
                     #cbar_kws={'label': clbl})# 'fontsize': 20})
        # Title
        title = f'Eigenmatch {ss+1} sim={similarities[0]:.2f}'
        ax.set_title(title, fontsize=12)

    #rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Learning curves
    if flg == 1:
        fig_learning_curves()

    # PCA variaince
    if flg == 2:
        #fig_pca(show_cum_point=0.99)
        fig_pca(outfile='fig_pca_noise.png',
            datasets=['MODIS_SST', 'MODIS_SST_2km', 'LLC_SST_nonoise', 'LLC_SST_noise'],
            show_cum_point=0.99)

    # Eigenmodes
    if flg == 3:
        fig_eigenimages('MNIST', 'Greys')

    # Eigenmodes
    if flg == 4:
        fig_eigenmatches('MODIS_SST', 'jet')


    # SWOT learning curve
    if flg == 30:
        #fig_learning_curve('SWOT')
        #fig_learning_curve('VIIRS')
        #fig_learning_curve('MODIS')
        fig_learning_curve('MNIST')

    # SWOT UMAP gallery
    if flg == 40:
        fig_swot_umap()


    # GRHSST talk
    if flg == 50:
        pass
        #fig_pca(outfile='fig_pca_MODIS.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MMV.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km', 'VIIRS_SST_2km'],
        #    show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MODIS_cumul.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    cumulative=True)
        #fig_pca(outfile='fig_pca_MODIS_frac.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    frac_remain=True)
        # MNIST + MODIS
        #fig_pca(outfile='fig_pca_MM.png',
        #    datasets=['MODIS_SST', 'MNIST'])
        # MNIST + MODIS + ImageNet
        #fig_pca(outfile='fig_pca_MMI.png',
        #    datasets=['MODIS_SST', 'MNIST',
        #              'ImageNet'])
        # MODIS + VIIRS
        #fig_pca(outfile='fig_pca_MV.png',
        #    datasets=['MODIS_SST', 'VIIRS'])
        # MODIS + VIIRS + LLC
        #fig_pca(outfile='fig_pca_MVL.png',
        #    datasets=['MODIS_SST', 'VIIRS',
        #              'LLC_SST'])
        # MODIS + VIIRS + LLC + SWOT
        #fig_pca(outfile='fig_pca_MVLS.png',
        #    datasets=['MODIS_SST', 'VIIRS',
        #              'LLC_SST', 'SWOT_L3'])
    


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)