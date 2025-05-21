""" Plotting routines """

import h5py
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from wrangler import s3_io
from remote_sensing.plotting import utils as plotting

from IPython import embed


def fig_learn_curve(valid_losses_file:str, train_losses_file:str, 
                    ylog:bool=False, outfile:str='fig_learn_curve.png'):
    # Grab the data
    #embed(header='17 of plotting')
    with s3_io.open(valid_losses_file, 'rb') as f:
        valid_hf = h5py.File(f, 'r')
        loss_valid = valid_hf['loss_valid'][:]
    #valid_hf.close()

    with s3_io.open(train_losses_file, 'rb') as f:
        train_hf = h5py.File(f, 'r')
        loss_train = train_hf['loss_train'][:]
    train_hf.close()

    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    ax.plot(np.arange(loss_valid.size)+1, loss_valid, label='validation', lw=3)
    ax.plot(np.arange(loss_train.size)+1, loss_train, c='red', label='training', lw=3)

    ax.legend(fontsize=23.)

    # Label
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if ylog:
        ax.set_yscale('log')

    plotting.set_fontsize(ax, 21.)
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))