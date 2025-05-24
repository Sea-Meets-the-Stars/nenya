""" Module to extract the information content of MODIS 2021 data """

import os
import numpy as np

from nenya import io as nenya_io
import extract_utils

from IPython import embed

local_modis_path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2')
modis_path = 's3://modis-l2'
extract_path = os.path.join(modis_path, 'Extractions')
local_extract_path = os.path.join(local_modis_path, 'Extractions')
local_preproc_path = os.path.join(local_modis_path, 'Info', 'PreProc')
local_info_path = os.path.join(local_modis_path, 'Info')
local_tables_path = os.path.join(local_modis_path, 'Info', 'Tables')
local_top_tables_path = os.path.join(local_modis_path, 'Tables')
preproc_path = os.path.join(modis_path, 'PreProc')
tables_path = os.path.join(modis_path, 'Tables')

tbl_2021_file = os.path.join(local_top_tables_path, 'modis_2021_128x128.parquet') 

def prep_2021_tbl():        

    # Orig table file
    orig_tbl_file = os.path.join(tables_path, 'MODIS_L2_20202021.parquet')
    tbl = nenya_io.load_main_table(orig_tbl_file, verbose=True)

    in_2021 = tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2021_95clear_128x128_preproc_standard.h5'

    # Cut
    tbl_2021 = tbl[in_2021].copy()
    # Reset index
    tbl_2021.reset_index(drop=True, inplace=True)
    # Write
    nenya_io.write_main_table(tbl_2021, tbl_2021_file, to_s3=False)

     

# Command line execution
if __name__ == '__main__':

    # Prep 2021 table
    #prep_2021_tbl()

    # Prep for Training
    if True:
        # Pre-process
        poptions={
            'de_mean': True,
            'median': True, 
            'med_size': (3,1),
            }
        # Go
        extract_utils.prep_for_training(tbl_2021_file,
                      os.path.join(local_extract_path, 'MODIS_R2019_2021_95clear_128x128_inpaint.h5'),
                      os.path.join(local_preproc_path, 'train_MODIS_2021_128x128.h5'), 
                      os.path.join(local_tables_path, 'train_MODIS_2021_128x128.parquet'), 
                      n_train=150000, n_valid=50000, poptions=poptions)#, debug=True)