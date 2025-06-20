""" Module to extract a training set for the LLC """

import os

import extract_utils

from IPython import embed

import info_defs

pdict = info_defs.grab_paths('LLC_SST')

local_llc_path = os.path.join(os.getenv('OS_OGCM'), 'LLC')
local_tables_path = os.path.join(local_llc_path, 'Info', 'Tables')
local_orig_preproc_path = os.path.join(local_llc_path, 'Nenya', 'PreProc')
local_preproc_path = os.path.join(local_llc_path, 'Info', 'PreProc')
tables_path = os.path.join(local_llc_path, 'Tables')

def ex_nonoise():
    # Open the LLC Uniform file
    poptions=None
    extract_utils.prep_for_training(os.path.join(tables_path, 'LLC_uniform144_r0.5_nonoise.parquet'),
                    os.path.join(local_orig_preproc_path, 'LLC_uniform144_nonoise_preproc.h5'),
                    os.path.join(local_preproc_path, 'train_llc_nonoise.h5'), 
                    os.path.join(local_tables_path, 'train_llc_nonoise.parquet'), 
                    inpaint=False, poptions=poptions,
                    use_ppidx=True, 
                    n_train=150000, n_valid=50000,
                    orig_key='valid')

# Command line execution
if __name__ == '__main__':
    ex_nonoise()