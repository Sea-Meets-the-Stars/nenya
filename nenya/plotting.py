""" Plotting routines """

import h5py
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from remote_sensing.plotting import utils as plotting
from wrangler import s3_io

from nenya import io as nenya_io

from IPython import embed

def closest_latents(images:list, similarities:np.ndarray, output_png:str='closest_latents.png'):

    # --- Shared color scale for all plots ---
    all_data = np.array(images)
    vmin = np.min(all_data)
    vmax = np.max(all_data)

    nimgs = all_data.shape[0]

    # --- Plot using pcolor ---
    plt.figure(figsize=(3 *nimgs, 3))

    for ss in range(nimgs):
        arr = images[ss]

        plt.subplot(1, nimgs, ss + 1)
        plt.pcolor(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.gca().set_aspect("auto")

        if ss == 0:
            plt.title("Query")
        else:
            plt.title(f"Sim: {similarities[ss]:.2f}")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"✅ Saved plot with pcolor to: {output_png}")


def learn_curve(valid_losses_file:str, train_losses_file:str, 
                ax=None, ylog:bool=False, outfile:str='fig_learn_curve.png'):

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
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        ax = plt.gca()

    ax.plot(np.arange(loss_valid.size)+1, loss_valid, label='validation', lw=3)
    ax.plot(np.arange(loss_train.size)+1, loss_train, c='red', label='training', lw=3)

    ax.legend(fontsize=23.)

    # Label
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if ylog:
        ax.set_yscale('log')

    plotting.set_fontsize(ax, 21.)
    
    if outfile is not None:
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print('Wrote {:s}'.format(outfile))
    


def umap_gallery(
    table_file:str, 
    image_file:str,
    umap_keys:list=['US0', 'US1'],
    img_key:str='valid',
    img_idx_key:str='index',
    xmnx:tuple=None,
    ymnx:tuple=None,
    dxv:float=0.5,
    dyv:float=0.25, 
    in_vmnx=None,
    scl_inset:tuple=(0.9,0.9),
    cbar_lbl:str='Field',
    outfile='fig_umap_gallery.png',
                     cmap='jet',
                     min_pts=None,
                     seed=None,
                     annotate=False,
                     use_std_lbls=True,
                     cut_to_inner:int=None,
                     skip_incidence=False,
                     debug=False): 
    """ UMAP gallery

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_umap_LL.png'.
        version (int, optional): [description]. Defaults to 1.
        local (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.
        cut_to_inner (int, optional): If provided, cut the image
            down to the inner npix x npix with npix = cut_to_inner

    Raises:
        IOError: [description]
    """
    if min_pts is None: 
        min_pts = 10
    # Seed
    if seed is not None:
        np.random.seed(seed)

    # Load table and unpack
    tbl = nenya_io.load_main_table(table_file)
    U0 = tbl[umap_keys[0]].values
    U1 = tbl[umap_keys[1]].values

    # Add index
    if img_idx_key not in tbl.keys() and img_idx_key == 'index':
        tbl[img_idx_key] = tbl.index.values

    # Open up the image file
    fimg = h5py.File(image_file, 'r')

    if debug:
        nxy = 4

    # Cut table

    # Range
    if xmnx is None:
        xmnx = (U0.min(), U0.max())
    if ymnx is None:
        ymnx = (U1.min(), U1.max())
    xmin, xmax = xmnx
    ymin, ymax = ymnx
        
    # cut
    good = (U0 > xmin) & (U0 < xmax) & (U1 > ymin) & (U1 < ymax)
    tbl = tbl.loc[good].copy()
    U0 = tbl[umap_keys[0]].values
    U1 = tbl[umap_keys[1]].values
    num_samples = len(tbl)
    print(f"We have {num_samples} making the cuts.")

    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        tbl = tbl.iloc[idx].copy()

    # Fig
    #_, cm = plotting.load_palette()
    fsz = 15.
    if annotate or skip_incidence:
        fsize = (9,8)
    else:
        fsize = (12,8)
    fig = plt.figure(figsize=fsize)
    plt.clf()

    if annotate:
        ax_gallery = fig.add_axes([0.10, 0.12, 0.75, 0.85])
    elif skip_incidence:
        ax_gallery = fig.add_axes([0.10, 0.12, 0.75, 0.85])
    else:
        ax_gallery = fig.add_axes([0.05, 0.1, 0.6, 0.90])

    if use_std_lbls:
        ax_gallery.set_xlabel(r'$U_0$')
        ax_gallery.set_ylabel(r'$U_1$')
    else:
        ax_gallery.set_xlabel(r'$'+umap_keys[0]+'$')
        ax_gallery.set_ylabel(r'$'+umap_keys[1]+'$')

    # Gallery
    ax_gallery.set_xlim(xmin, xmax)
    ax_gallery.set_ylim(ymin, ymax)

    print('x,y', xmin, xmax, ymin, ymax, dxv, dyv)

    
    # ###################
    # Gallery time

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # Ugly for loop
    ndone = 0
    if debug:
        nmax = 100
    else:
        nmax = 1000000000

    # Color bar
    plt_cbar = True
    ax_cbar = ax_gallery.inset_axes(
                    [xmax + dxv/10, ymin, dxv/2, (ymax-ymin)*0.2],
                    transform=ax_gallery.transData)
    cbar_kws = dict(label=cbar_lbl)#r'SSTa (K)')

    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((U0 >= x) & (
                U0 < x+dxv) & (
                U1 >= y) & (U1 < y+dxv))[0]
            if len(pts) < min_pts:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = tbl.iloc[idx]

            # Image
            axins = ax_gallery.inset_axes(
                    [x, y, scl_inset[0]*dxv, scl_inset[1]*dyv], 
                    transform=ax_gallery.transData)
            # Load
            #embed(header='205 of plotting')                                                    
            try:
                cutout_img = fimg[img_key][cutout[img_idx_key]]
            except:
                embed(header='631 of plotting')                                                    
            if cutout_img.ndim == 3:
                cutout_img = cutout_img[0,:,:]
            '''
            # Cut down?
            if cut_to_inner is not None:
                imsize = cutout_img.shape[0]
                x0, y0 = [imsize//2-cut_to_inner//2]*2
                x1, y1 = [imsize//2+cut_to_inner//2]*2
                cutout_img = cutout_img[x0:x1,y0:y1]
            '''
            # Limits
            if in_vmnx is not None:
                vmnx = in_vmnx
            else:
                imin, imax = cutout_img.min(), cutout_img.max()
                amax = max(np.abs(imin), np.abs(imax))
                vmnx = (-1*amax, amax)
            # Plot
            sns_ax = sns.heatmap(np.flipud(cutout_img), 
                            xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cmap, cbar=plt_cbar,
                     cbar_ax=ax_cbar, cbar_kws=cbar_kws,
                     ax=axins)
            sns_ax.set_aspect('equal', 'datalim')
            # Only do this once
            if plt_cbar:
                plt_cbar = False
            ndone += 1
            print(f'ndone= {ndone}')
            if ndone > nmax:
                break
        if ndone > nmax:
            break

    plotting.set_fontsize(ax_gallery, fsz)
    #ax.set_aspect('equal', 'datalim')
    #ax.set_aspect('equal')#, 'datalim')

    '''
    # Box?
    if umap_rngs is not None:
        umap_rngs = parse_umap_rngs(umap_rngs)
            # Create patch collection with specified colour/alpha
        rect = Rectangle((umap_rngs[0][0], umap_rngs[1][0]),
            umap_rngs[0][1]-umap_rngs[0][0],
            umap_rngs[1][1]-umap_rngs[1][0],
            linewidth=2, edgecolor='k', facecolor='none', ls='-',
            zorder=10)
        ax_gallery.add_patch(rect)

    # Another?
    if extra_umap_rngs is not None:
        umap_rngs = parse_umap_rngs(extra_umap_rngs)
            # Create patch collection with specified colour/alpha
        rect2 = Rectangle((umap_rngs[0][0], umap_rngs[1][0]),
            umap_rngs[0][1]-umap_rngs[0][0],
            umap_rngs[1][1]-umap_rngs[1][0],
            linewidth=2, edgecolor='k', facecolor='none', ls='--',
            zorder=10)
        ax_gallery.add_patch(rect2)

    # Incidence plot
    if not annotate and not skip_incidence:
        ax_incidence = fig.add_axes([0.71, 0.45, 0.25, 0.36])

        fig_umap_density(outfile=None, modis_tbl=modis_tbl,
                     umap_grid=umap_grid, umap_comp=umap_comp,
                     show_cbar=True, ax=ax_incidence, fsz=12.)
    #ax_incidence.plot(np.arange(10), np.arange(10))
    '''

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))