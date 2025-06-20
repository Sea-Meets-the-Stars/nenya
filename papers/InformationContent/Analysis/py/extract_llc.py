""" Module to extract a training set for the LLC """

import os

import extract_utils

from IPython import embed

import info_defs

pdict = info_defs.grab_paths('LLC_SST')

local_llc_path = os.path.join(os.getenv('OS_OGCM'), 'LLC')
local_tables_path = os.path.join(local_llc_path, 'Info', 'Tables')
tables_path = os.path.join(local_llc_path, 'Tables')

def main():
    # Open the LLC Uniform file
    poptions=None
    extract_utils.prep_for_training(os.path.join(tables_path, 'VIIRS_N21_2024.parquet'), 
                    os.path.join(local_extract_path, 'ex_VIIRS_N21_2024.h5'),
                    os.path.join(local_preproc_path, 'train_VIIRS_N21_2024.h5'), 
                    os.path.join(local_tables_path, 'train_VIIRS_N21_2024.parquet'), 
                    inpaint=False, poptions=poptions,
                    n_train=150000, n_valid=50000)

# Command line execution
if __name__ == '__main__':
    main()