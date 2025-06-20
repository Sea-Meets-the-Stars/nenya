""" Module to extract the information content of the 2024 data """

import os
import numpy as np

from wrangler.extract.grab_and_go import run as gg_run

import extract_utils

from IPython import embed

tstart = '2024-01-01T00:00:00'
tend = '2024-12-31T23:59:59'

local_viirs_path = os.path.join(os.getenv('OS_SST'), 'VIIRS')
viirs_path = 's3://viirs'
extract_path = os.path.join(viirs_path, 'Extractions')
local_extract_path = os.path.join(local_viirs_path, 'Extractions')
local_preproc_path = os.path.join(local_viirs_path, 'Info', 'PreProc')
local_info_path = os.path.join(local_viirs_path, 'Info')
local_tables_path = os.path.join(local_viirs_path, 'Info', 'Tables')
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

        
     

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Learning curves
    if flg == 1:
        # Extract N21
        extract_viirs('VIIRS_N21', 'extract_viirs_std.json', 
                    'ex_VIIRS_N21_2024.h5', 'VIIRS_N21_2024.parquet',
                    n_cores=15)#, debug=True)#, debug_async=True, debug=True)
                    #n_cores=15, debug=True)

    # Pre-process at native resolution
    if flg == 2:
        poptions={
            'de_mean': True,
            'median': False, 
            }
        extract_utils.prep_for_training(os.path.join(tables_path, 'VIIRS_N21_2024.parquet'), 
                        os.path.join(local_extract_path, 'ex_VIIRS_N21_2024.h5'),
                        os.path.join(local_preproc_path, 'train_VIIRS_N21_2024.h5'), 
                        os.path.join(local_tables_path, 'train_VIIRS_N21_2024.parquet'), 
                        inpaint=True, poptions=poptions,
                        n_train=150000, n_valid=50000)#, debug=True)

    # Same sample but downscale 3x3 to 64^2
    if flg == 3:
        extract_utils.downscale(
            os.path.join(local_preproc_path, 'train_VIIRS_N21_2024.h5'),
            os.path.join(local_preproc_path, 'train_VIIRS_N21_2024_2km.h5'),
            dscale_size=(3, 3), n_cores=15)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)