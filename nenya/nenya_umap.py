""" UMAP utitlies for Nenya """
import numpy as np
import os, shutil
import glob
import pickle

import h5py
import pandas
import umap

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
#from ulmo.nenya.train_util import option_preprocess
#from ulmo.nenya import defs as ssl_defs

from nenya import params
from nenya import defs

from IPython import embed

def DT_interval(inp:tuple):
    """ Generate a DT interval from the input

    Args:
        inp (tuple or None): DT central value and dDT

    Returns:
        tuple: Range of DT values
    """
    if inp is None: # All
        return (0., 1e12)
    DT, dDT = inp
    if dDT < 0:
        return (DT, DT+1e12)
    else:
        return (DT-dDT, DT+dDT)

def load(model_name:str, DT:float=None, use_s3:bool=False):
    """ Load a UMAP model

    Args:
        model_name (str): 
            Model name ['LLC', 'LLC_local', 'CF', 'v4', 'v5']
        DT (float, optional):
            DT value (K). Defaults to None. 
        use_s3 (bool, optional): 
            Use s3? Defaults to False.

    Raises:
        IOError: _description_
        IOError: _description_
        IOError: _description_

    Returns:
        tuple: UMAP model, table file
    """
    tbl_file = None
    if model_name == 'LLC':
        umap_file = 's3://llc/SSL/LLC_MODIS_2012_model/ssl_LLC_v1_umap.pkl'
    elif model_name == 'LLC_local':
        umap_file = './ssl_LLC_v1_umap.pkl'
    elif model_name in ['CF', 'v4', 'v5', 'viirs_v1']: 
        # Root
        sensor = 'VIIRS' if model_name == 'viirs_v1' else 'MODIS'
        tbl_pth = 'VIIRS' if model_name == 'viirs_v1' else 'MODIS_L2'
        if model_name == 'CF':
            umap_root = 'cloud_free'
            nname = 'SSL'
        elif model_name == 'v4':  # Probably lost to the vaguries of UMAP pickeling
            umap_root = '96clear_v4'
            nname = 'SSL'
        elif model_name == 'v5':  # Probably lost to the vaguries of UMAP pickeling
            umap_root = '96clear_v5'
            nname = 'Nenya'
        elif model_name == 'viirs_v1':  # VIIRS model
            umap_root = 'v1'
            nname = 'Nenya_98clear'
        else:
            raise IOError("Bad model_name")

        if use_s3:
            raise IOError("Not ready for s3!")
        else:
            umap_path = os.path.join(os.getenv('OS_SST'),
                                 tbl_pth, 'Nenya', 'UMAP')
        
        for key in ssl_defs.umap_DT.keys():
            if key in ['all', 'DT10']:
                continue
            DT_rng = DT_interval(ssl_defs.umap_DT[key])
            if DT_rng[0] < DT <= DT_rng[1]:
                umap_file = os.path.join(
                    umap_path, 
                    f'{sensor}_{nname}_{umap_root}_{key}_UMAP.pkl')
                                    
        tbl_file = os.path.join(
            os.getenv('OS_SST'), tbl_pth, 'Nenya', 'Tables', 
            os.path.basename(umap_file).replace(
                '_UMAP.pkl', '.parquet'))
    else:
        raise IOError("bad model")

    # Download?
    if use_s3:
        umap_base = os.path.basename(umap_file)
        if not os.path.isfile(umap_base):
            ulmo_io.download_file_from_s3(umap_base, umap_file)
    else: # local
        umap_base = umap_file
    print(f"Loading UMAP: {umap_base}")

    # Return
    return pickle.load(ulmo_io.open(umap_base, "rb")), tbl_file

def umap_subset(tbl:pandas.DataFrame, 
                opt_path:str, outfile:str, 
                DT_cut:str=None, 
                alpha_cut:str=None, 
                max_cloud_fraction:float=None,
                ntrain=200000, remove=True,
                DT_key:str='DT40',
                umap_savefile:str=None,
                train_umap:bool=True,
                local=True, CF=False, debug=False):
    """Run UMAP on a subset of the data
    First 2 dimensions are written to the table

    Args:
        tbl (pandas.DataFrame): Data table
        opt_path (str): _description_
        outfile (str): _description_
        DT_cut (str, optional): DT cut to apply. Defaults to None.
        alpha_cut (str, optional): alpha cut to apply. Defaults to None.
        ntrain (int, optional): _description_. Defaults to 200000.
        remove (bool, optional): _description_. Defaults to True.
        umap_savefile (str, optional): _description_. Defaults to None.
        local (bool, optional): _description_. Defaults to True.
        CF (bool, optional): Use cloud free (99%) set? Defaults to False.
        train_umap (bool, optional): Train a new UMAP? Defaults to True.
        debug (bool, optional): _description_. Defaults to False.

    Raises:
        IOError: _description_
        IOError: _description_
    """
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    '''
    # Load main table
    table='CF' if CF else '96'
    tbl['US0'] = 0.
    tbl['US1'] = 0.
    '''

    # Cut down on DT
    if DT_cut is not None:
        DT_cuts = defs.umap_DT[DT_cut]
        if DT_cuts[1] < 0: # Lower limit?
            keep = tbl[DT_key] > DT_cuts[0]
        else:
            keep = np.abs(tbl[DT_key] - DT_cuts[0]) < DT_cuts[1]
    else: # Do em all! (or cut on alpha)
        keep = np.ones(len(tbl), dtype=bool)

    # Cut down on clouds?
    if max_cloud_fraction is not None:
        keep = keep & (tbl.clear_fraction < max_cloud_fraction)

    # Cut down on alpha
    if DT_cut is None and alpha_cut is not None:
        alpha_cuts = defs.umap_alpha[alpha_cut]
        if alpha_cuts[1] < 0: # Upper limit?
            keep = tbl.min_slope < alpha_cuts[0]
        else:
            keep = np.abs(tbl.min_slope - alpha_cuts[0]) < alpha_cuts[1]

    tbl = tbl[keep].copy()
    print(f"After the cuts, we have {len(tbl)} cutouts to work on.")

    # Ulmo pp_type
    valid = tbl.ulmo_pp_type == 0
    train = tbl.ulmo_pp_type == 1
    cut_prefix = 'ulmo_'

    # Prep latent_files
    print(f"Loading latents from this folder: {opt.latents_folder}")
    if local:
        latents_path = os.path.join(
            os.getenv('OS_SST'), opt.sensor, 'Nenya', opt.latents_folder)
        latent_files = glob.glob(latents_path)
    else:
        latents_path = os.path.join(opt.s3_outdir, opt.latents_folder)
        latent_files = ulmo_io.list_of_bucket_files(latents_path)
        latent_files = [opt.s3_bucket+item for item in latent_files]

    # Load up all the latent vectors

    # Loop on em all
    if debug:
        latent_files = latent_files[0:2]

    if debug:
        embed(header='178 of umap')

    all_latents = []
    sv_idx = []
    for latents_file in latent_files:
        basefile = os.path.basename(latents_file)

        pfile = basefile.replace('_latents', '_preproc')
        yidx = tbl.pp_file == f'{opt.s3_bucket}PreProc/{pfile}'

        if debug:
            embed(header='223 of umap')

        # Download?
        if not os.path.isfile(basefile):
            # Try local if local is True
            if local:
                shutil.copyfile(latents_file, basefile)
            if not os.path.isfile(basefile):
                print(f"Downloading {latents_file} (this is *much* faster than s3 access)...")
                ulmo_io.download_file_from_s3(basefile, latents_file)

        #  Load and apply
        print(f"Ingesting {basefile}")
        hf = h5py.File(basefile, 'r')

        # ##############33
        # Valid
        all_latents_valid = hf['valid'][:]

        valid_idx = valid & yidx
        pp_idx = tbl[valid_idx].pp_idx.values

        # Grab and save
        gd_latents = all_latents_valid[pp_idx, :]
        all_latents.append(gd_latents)
        sv_idx += np.where(valid_idx)[0].tolist()

        # Train
        train_idx = train & yidx
        if 'train' in hf.keys() and (np.sum(train_idx) > 0):
            pp_idx = tbl[train_idx].pp_idx.values
            all_latents_train = hf['train'][:]
            gd_latents = all_latents_train[pp_idx, :]
            all_latents.append(gd_latents)
            sv_idx += np.where(train_idx)[0].tolist()

        hf.close()
        # Clean up
        if not debug and remove:
            print(f"Done with {basefile}.  Cleaning up")
            os.remove(basefile)

    # Concatenate
    all_latents = np.concatenate(all_latents, axis=0)
    nlatents = all_latents.shape[0]

    # UMAP me
    if debug:
        embed(header='218 of umap')

    if train_umap:
        ntrain = min(ntrain, nlatents)
        print(f"Training UMAP on a random {ntrain} set of the files")
        random = np.random.choice(np.arange(nlatents), size=ntrain, 
                                replace=False)
        reducer_umap = umap.UMAP(random_state=42)
        latents_mapping = reducer_umap.fit(all_latents[random,...])
        print("Done..")

        # Save?
        if umap_savefile is not None:
            pickle.dump(latents_mapping, open(umap_savefile, "wb" ) )
            print(f"Saved UMAP to {umap_savefile}")
    else:
        latents_mapping = pickle.load(open(umap_savefile, "rb" ) )
        print(f"Loaded UMAP from {umap_savefile}")


    # Evaluate all of the latents
    print("Embedding all of the latents..")
    embedding = latents_mapping.transform(all_latents)

    # Save
    srt = np.argsort(sv_idx)
    gd_idx = np.zeros(len(tbl), dtype=bool)
    gd_idx[sv_idx] = True
    tbl.US0.values[gd_idx] = embedding[srt,0]
    tbl.US1.values[gd_idx] = embedding[srt,1]
    #tbl.loc[gd_idx, 'US0'] = embedding[srt,0]
    #tbl.loc[gd_idx, 'US1'] = embedding[srt,1]

    # Remove DT
    drop_columns = []
    for key in ['DT', 'logDT', 'lowDT', 'absDT', 'min_slope']: 
        if key in tbl.keys():
            drop_columns.append(key)
    if len(drop_columns) > 0:
        tbl.drop(columns=drop_columns, inplace=True)
    
    # Vet
    assert cat_utils.vet_main_table(tbl, cut_prefix=cut_prefix)
    # Write new table
    to_s3 = True if 's3' in outfile else False
    if not debug:
        ulmo_io.write_main_table(tbl, outfile, to_s3=to_s3)

def grid_umap(U0:np.ndarray, U1:np.ndarray, nxy:int=16, 
              percent:list=[0.05, 99.95], verbose=False):
    """ 
    Generate a grid on the UMAP domain
    """

    # Boundaries of the grid
    xmin, xmax = np.percentile(U0, percent)
    ymin, ymax = np.percentile(U1, percent)
    dxv = (xmax-xmin)/nxy
    dyv = (ymax-ymin)/nxy

    if verbose:
        print(f"DU_0={dxv} and DU_1={dyv}")

    # Edges
    xmin -= dxv
    xmax += dxv
    ymin -= dyv
    ymax += dyv

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # Return
    grid = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                xval=xval, yval=yval, dxv=dxv, dyv=dyv)
    return grid


def cutouts_on_umap_grid(tbl:pandas.DataFrame, nxy:int, 
                         umap_keys:tuple, min_pts:int=1):
    """ Genreate a list of cuotuts uniformly distributed on the UMAP grid
    """

    # Grid
    umap_grid = grid_umap(
        tbl[umap_keys[0]].values,
        tbl[umap_keys[1]].values, 
        nxy=nxy)

    # Unpack
    xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
    ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
    dxv = umap_grid['dxv']
    dyv = umap_grid['dyv']

    # Cut
    good = (tbl[umap_keys[0]] > xmin) & (
        tbl[umap_keys[0]] < xmax) & (
        tbl[umap_keys[1]] > ymin) & (
            tbl[umap_keys[1]] < ymax) & np.isfinite(tbl.LL)

    tbl = tbl.loc[good].copy()
    num_samples = len(tbl)
    print(f"We have {num_samples} making the cuts.")

    # Grid
    xval = umap_grid['xval']
    yval = umap_grid['yval']

    # Grab cutouts
    cutouts = []
    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((tbl[umap_keys[0]] >= x) & (
                tbl[umap_keys[0]] < x+dxv) & (
                tbl[umap_keys[1]] >= y) & (tbl[umap_keys[1]] < y+dxv)
                           & np.isfinite(tbl.LL))[0]
            if len(pts) < min_pts:
                cutouts.append(None)
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = tbl.iloc[idx]
            # Save
            cutouts.append(cutout)

    # Return
    return tbl, cutouts, umap_grid


def regional_analysis(geo_region:str, tbl:pandas.DataFrame, nxy:int, 
                      umap_keys:tuple, min_counts:int=200):
    # Grid
    grid = grid_umap(tbl[umap_keys[0]].values, 
        tbl[umap_keys[1]].values, nxy=nxy)
 
    # cut
    good = (tbl[umap_keys[0]] > grid['xmin']) & (
        tbl[umap_keys[0]] < grid['xmax']) & (
        tbl[umap_keys[1]] > grid['ymin']) & (
            tbl[umap_keys[1]] < grid['ymax']) & np.isfinite(tbl.LL)

    tbl = tbl.loc[good].copy()
    num_samples = len(tbl)
    print(f"We have {num_samples} making the cuts.")

    # All
    counts, xedges, yedges = np.histogram2d(
        tbl[umap_keys[0]], 
        tbl[umap_keys[1]], bins=(grid['xval'], 
        grid['yval']))

    # Normalize
    if min_counts > 0:
        counts[counts < min_counts] = 0.
    counts /= np.sum(counts)

    # Geographic
    lons = defs.geo_regions[geo_region]['lons']
    lats = defs.geo_regions[geo_region]['lats']
    #embed(header='739 of figs')
    geo = ( (tbl.lon > lons[0]) &
        (tbl.lon < lons[1]) &
        (tbl.lat > lats[0]) &
        (tbl.lat < lats[1]) )

    geo_tbl = tbl.loc[good & geo].copy()
    counts_geo, xedges, yedges = np.histogram2d(
        geo_tbl[umap_keys[0]], 
        geo_tbl[umap_keys[1]], bins=(grid['xval'], 
                                     grid['yval']))
    print(f"There are {len(geo_tbl)} cutouts in the geographic region")

    # Normalize
    counts_geo /= np.sum(counts_geo)

    # Ratio
    rtio_counts = counts_geo / counts

    # Return
    return counts, counts_geo, tbl, grid, xedges, yedges


def old_latents_umap(latents:np.ndarray, train:np.ndarray, 
         valid:np.ndarray, valid_tbl:pandas.DataFrame,
         fig_root='', transformer_file=None):
    """ Run a UMAP on input latent vectors.
    A subset are used to train the UMAP and then
    one applies it to the valid set.

    The UMAP U0, U1 coefficients are written to an input table.

    Args:
        latents (np.ndarray): Total set of latent vectors (training+valid)
            Shape should be (nvectors, size of latent space)
        train (np.ndarray): indices for training
        valid (np.ndarray): indices for applying the UMAP
        valid_tbl (pandas.DataFrame): [description]
        fig_root (str, optional): [description]. Defaults to ''.
        transformer_file (str, optional): Write the UMAP fit to this file
    """
    raise ValueError("This is old code. Deprecate it")  

    # UMAP me
    print("Running UMAP..")
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(latents[train])
    if transformer_file is not None:
        pickle.dump(latents_mapping, open(transformer_file, "wb" ) )
        tmp = pickle.load(ulmo_io.open(transformer_file, "rb" ) )
    print("Done")

    # Apply to embedding
    print("Applying to the valid images")
    train_embedding = latents_mapping.transform(latents[train])
    valid_embedding = latents_mapping.transform(latents[valid])
    print("Done")

    # Quick figures
    if len(fig_root) > 0:
        print("Generating plots")
        num_samples = train.size
        point_size = 20.0 / np.sqrt(num_samples)
        dpi = 100
        width, height = 800, 800

        plt.figure(figsize=(width//dpi, height//dpi))
        plt.scatter(latents_mapping.embedding_[:, 0], 
            latents_mapping.embedding_[:, 1], s=point_size)
        plt.savefig(fig_root+'_train_UMAP.png')

        # New plot
        num_samplesv = latents[valid].shape[0]
        point_sizev = 1.0 / np.sqrt(num_samplesv)
        plt.figure(figsize=(width//dpi, height//dpi))
        ax = plt.gca()
        img = ax.scatter(valid_embedding[:, 0], 
                valid_embedding[:, 1], s=point_sizev,
            c=valid_tbl.LL, cmap='jet', vmin=-1000)
        cb = plt.colorbar(img, pad=0.)
        cb.set_label('LL', fontsize=20.)
        #
        ax.set_xlabel(r'$U_0$')
        ax.set_ylabel(r'$U_1$')
        plotting.set_fontsize(ax, 15.)
        #
        plt.savefig(fig_root+'_valid_UMAP.png', dpi=300)

    # Return
    return train_embedding, valid_embedding, latents_mapping