""" Module to extract the information content of the 2024 data """

import os

import numpy as np
import pandas
import h5py

from wrangler.extract.grab_and_go import run as gg_run
from wrangler import io as wr_io

from nenya import io as nenya_io

from IPython import embed

tstart = '2024-01-01T00:00:00'
tend = '2024-12-31T23:59:59'

local_viirs_path = os.path.join(os.getenv('OS_SST'), 'VIIRS')
viirs_path = 's3://viirs'
extract_path = os.path.join(viirs_path, 'Extractions')
local_extract_path = os.path.join(local_viirs_path, 'Extractions')
preproc_path = os.path.join(viirs_path, 'PreProc')
tables_path = os.path.join(viirs_path, 'Tables')

def extract_viirs(dataset:str, eoption_file:str,
                  ex_file:str, tbl_file:str, n_cores:int=15,
                  debug=False):

    gg_run(dataset, tstart, tend, eoption_file, 
                       ex_file, tbl_file, n_cores, 
                       verbose=True, debug=debug, 
                       save_local_files=True,
                       debug_noasync=debug)

def prep_for_training(tbl_file:str, 
                      extract_file:str, 
                      preproc_file:str, 
                      n_train:int=300000, 
                      n_valid:int=100000,
                      debug:bool=False):
    """Prepare the data for training

    Args:
        tbl_file (str): Parquet file with the data
        train_file (str): Parquet file with the training data
        extract_file (str): HDF5 file with the extracted data
        preproc_file (str): HDF5 file with the pre-processed data
        n_train (int, optional): Number of training samples. Defaults to 300000.
        n_valid (int, optional): Number of validation samples. Defaults to 100000.
    """

    # Open the table
    df = nenya_io.load_main_table(tbl_file, verbose=True)

    # Random select n_train samples and n_valid samples
    idx_tv = np.random.choice(df.index, n_train+n_valid, replace=False)

    # Split into training and validation
    df['pp_file'] = preproc_file
    df['pp_type'] = -1
    # Set train
    df.loc[idx_tv[:n_train], 'pp_type'] = 1
    # Set valid
    df.loc[idx_tv[n_train:], 'pp_type'] = 0
    # Indices
    df['pp_idx'] = -1
    df.loc[idx_tv[:n_train], 'pp_idx'] = np.arange(n_train)
    df.loc[idx_tv[n_train:], 'pp_idx'] = np.arange(n_valid)

    # Data time
    f = h5py.File(extract_file, 'r')
    base_preproc = os.path.basename(preproc_file)
    train_f = h5py.File(base_preproc, 'w')

    # Save the training and validation data
    if not debug:
        # Load h5
        fields = f['fields'][:]
        sub_fields = np.zeros((n_train+n_valid, fields.shape[1], fields.shape[2]), 
                              dtype=np.float32)
        for kk, idx in enumerate(idx_tv):
            sub_fields[kk] = fields[idx]
        del fields

        inpainted = f['inpainted_masks'][:]

        # Inpaint
        for kk, idx in enumerate(idx_tv):
            fill = np.isfinite(inpainted[idx])
            fields[kk][fill] = inpainted[idx][fill]

        del inpainted

        # Write
        train_f.create_dataset('train', data=fields[:n_train])
        train_f.create_dataset('valid', data=fields[n_train:])
        train_f.close()

        # Push to s3
        wr_io.upload_file_to_s3(base_preproc, preproc_file)

        # Table
        nenya_io.write_main_table(df, tbl_file, to_s3=True)
        
     

# Command line execution
if __name__ == '__main__':

    '''
    # Extract N21
    extract_viirs('VIIRS_N21', 'extract_viirs_std.json', 
                  'ex_VIIRS_N21_2024.h5', 'VIIRS_N21_2024.parquet',
                  n_cores=15)#, debug=True)#, debug_async=True, debug=True)
                  #n_cores=15, debug=True)
    '''

    # Prep for Training
    prep_for_training(os.path.join(tables_path, 'VIIRS_N21_2024.parquet'), 
                      os.path.join(local_extract_path, 'ex_VIIRS_N21_2024.h5'),
                      os.path.join(preproc_path, 'train_VIIRS_N21_2024.h5'), 
                      n_train=300000, n_valid=100000)#, debug=True)