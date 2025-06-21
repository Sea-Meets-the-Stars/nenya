
""" Utility functions for preparing data for training
"""
import os
import numpy as np
import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from skimage.transform import downscale_local_mean 

from wrangler.preproc import field as pp_field
from wrangler.plotting import cutout as plot_cutout
from nenya import io as nenya_io

from IPython import embed

def wrap_preproc_field(field, poptions:dict=None):
    """Wrapper for the preprocessing function to handle options"""
    return pp_field.main(field, None, **poptions)

def prep_for_training(tbl_file:str, 
                      extract_file:str, 
                      preproc_file:str, 
                      train_tbl_file:str=None,
                      inpaint:bool=False,
                      n_train:int=300000, 
                      n_valid:int=100000,
                      use_ppidx:bool=False,
                      debug:bool=False,
                      poptions:dict=None,
                      orig_key:str='fields',
                      n_cores:int=20):
    """Prepare the data for training

    Args:
        tbl_file (str): Parquet file with the data
        train_file (str): Parquet file with the training data
        extract_file (str): HDF5 file with the extracted data
        preproc_file (str): HDF5 file with the pre-processed data
        n_train (int, optional): Number of training samples. Defaults to 300000.
        n_valid (int, optional): Number of validation samples. Defaults to 100000.
        poptions (dict, optional): Preprocessing options. Defaults to None.
    """

    print(f"Preparing the data for training.  n_train={n_train}, n_valid={n_valid}")
    # Open the table
    df = nenya_io.load_main_table(tbl_file, verbose=True)

    # Cut on pp_idx?
    if use_ppidx:
        # Cut the table to the pp_idx
        df = df[df['pp_idx'] >= 0].copy()
        print(f"Cutting the table to {len(df)} samples with pp_idx >= 0")

    # Random select n_train samples and n_valid samples
    idx_tv = np.random.choice(df.index, n_train+n_valid, replace=False)

    if use_ppidx:
        ex_idx = df.loc[idx_tv, 'pp_idx'].values
    else:
        ex_idx = idx_tv
        

    # Split into training and validation
    df['pp_file'] = preproc_file
    df['pp_type'] = -1
    # Set train
    df.loc[idx_tv[:n_train], 'pp_type'] = 1
    # Set valid
    df.loc[idx_tv[n_train:], 'pp_type'] = 0

    # Indices (they will be aligned)
    df['pp_idx'] = -1
    df.loc[idx_tv[:n_train], 'pp_idx'] = np.arange(n_train)
    df.loc[idx_tv[n_train:], 'pp_idx'] = np.arange(n_valid)

    # Data time
    f = h5py.File(extract_file, 'r')
    train_f = h5py.File(preproc_file, 'w')

    # Save the training and validation data
    if not debug:
        # Load h5
        print("Loading the fields..")
        fields = f[orig_key][:]
        if fields.ndim == 4:
            # If the fields are 4D, we need to squeeze them
            fields = fields.squeeze(axis=1)
        sub_fields = np.zeros((n_train+n_valid, fields.shape[1], fields.shape[2]), 
                              dtype=np.float32)
        for kk, idx in enumerate(ex_idx):
            sub_fields[kk] = fields[idx]
        del fields

        # Inpaint
        if inpaint:
            print("Inpainting..")
            inpainted = f['inpainted_masks'][:]
            for kk, idx in enumerate(idx_tv):
                fill = np.isfinite(inpainted[idx])
                sub_fields[kk][fill] = inpainted[idx][fill]
            del inpainted

        # Preprocess?
        if poptions is not None:
            # Map me
            map_fn = partial(wrap_preproc_field, poptions=poptions)

            # Generate a list
            sub_fields = [sub_fields[i] for i in range(sub_fields.shape[0])]
            # Loop over the files and pre-process them all
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(sub_fields) // n_cores if len(sub_fields) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, sub_fields,
                                                    chunksize=chunksize), 
                                    total=len(sub_fields),
                                    desc='Preprocessing fields',
                                    unit='field'))
            # Unpack
            sub_fields = np.array([item[0] for item in answers], dtype=np.float32)
            del answers

        # Write
        print("Writing..")
        train_f.create_dataset('train', data=sub_fields[:n_train].astype(np.float32))
        train_f.create_dataset('valid', data=sub_fields[n_train:].astype(np.float32))
        train_f.close()
        print(f"Wrote {n_train} training and {n_valid} validation samples to {preproc_file}")

        #wr_io.upload_file_to_s3(base_preproc, preproc_file)

        # Table
        nenya_io.write_main_table(df, train_tbl_file, to_s3=False)

        # Push to s3
        print("Upload to s3 yourself!")

def downtime(field, dscale_size:tuple=(3,3)):
    """Downsample the field by a factor of dscale"""
    return downscale_local_mean(field, dscale_size)

def add_noise(field, noise:float=0.01):
    field += np.random.normal(loc=0., 
        scale=noise, 
        size=field.shape)
    return field

def modify(orig_file:str, new_file:str, map_fn,
              n_cores:int=15): 
    """Modify the original file by the map function

    Args:
        native_file (str): Path to the native resolution file
        new_file (str): Path to the new file
    """
    #map_fn = partial(downtime, dscale_size=dscale_size)

    print(f"Modifying {orig_file} to {new_file}")

    print("Loading")
    with h5py.File(orig_file, 'r') as f:
        train = f['train'][:]
        valid = f['valid'][:]
    print("Done")

    print("Lists")
    train = [item for item in train]
    valid = [item for item in valid]

    ntrain = len(train)
    print("Combine")
    fields = train + valid
    del train, valid

    # Downscale
    print("Downscale")
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(fields) // n_cores if len(fields) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, fields,
                                            chunksize=chunksize), 
                            total=len(fields),
                            desc='Preprocessing fields',
                            unit='field'))
    #embed(header='182 of downscale')
    train = np.array([item for item in answers[0:ntrain]], dtype=np.float32)
    valid = np.array([item for item in answers[ntrain:]], dtype=np.float32)
    
    # Save
    with h5py.File(new_file, 'w') as nf:
        nf.create_dataset('train', data=train)
        nf.create_dataset('valid', data=valid)
    print(f"Wrote downscaled data to {new_file}")