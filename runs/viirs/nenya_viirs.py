""" Nenya Analayis of VIIRS -- 
"""
import os
from typing import IO
import numpy as np

import h5py
import numpy as np
import argparse

import pandas
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from nenya import train as nenya_train
from nenya import latents_extraction
from nenya import io as nenya_io
from nenya import params 
from nenya import nenya_umap
from nenya import analyze_image

from IPython import embed


def train(opt_path:str, debug:bool=False, save_file:str=None):
    """Train the model

    Args:
        opt_path (str): Path + filename of options file
        debug (bool, optional): 
        save_file (str, optional): 
    """
    # Do it
    nenya_train.main(opt_path, debug=debug, save_file=save_file)
        

def prep_nenya_table(opt_path:str, debug=False):
    

    # Parse the model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    # Check for the Table
    chk_tbl_file = ulmo_io.list_of_bucket_files(opt.nenya_tbl_file)
    if len(chk_tbl_file) > 0:
        print(f"Table already exists: {opt.nenya_tbl_file}")
        return

    # Data files
    all_pp_files = ulmo_io.list_of_bucket_files('viirs', 'PreProc')
    pp_files = []
    for ifile in all_pp_files:
        if opt.eval_root in ifile:
            pp_files.append(f's3://viirs/{ifile}')

    # Table
    viirs_tbl = ulmo_io.load_main_table(opt.orig_tbl_file)
    viirs_tbl.rename(columns={'pp_type':'ulmo_pp_type',
                              'pp_idx':'ulmo_pp_idx', 
                              'pp_file':'ulmo_pp_file'}, 
                     inplace=True)

    for ifile in pp_files:
        print(f"Working on {ifile}")
        data_file = os.path.basename(ifile)

        # Download
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, ifile)
        else:
            print(f"Data file already downloaded: {data_file}")

        # T40
        print("Calculating DT40") 
        f = h5py.File(data_file, 'r')
        for itype in ['valid', 'train']:
            if itype not in f.keys():
                continue
            #
            print(f"Working on {itype}")
            images = f[itype][:]
            DT40s = analyze_image.calc_DT(
                images, opt.random_jitter, 
                verbose=False)
            # Fill
            ppt = 0 if itype == 'valid' else 1
            idx = (viirs_tbl.ulmo_pp_file == ifile) & (
                viirs_tbl.ulmo_pp_type == ppt)
            pp_idx = viirs_tbl[idx].ulmo_pp_idx.values
            viirs_tbl.loc[idx.values, 'DT40'] = DT40s[pp_idx]
        f.close()

        # Remove data file
        if not debug:
            os.remove(data_file)
            print(f'{data_file} removed')

    # Save the table
    assert cat_utils.vet_main_table(viirs_tbl, cut_prefix='ulmo_')
    if not debug:
        ulmo_io.write_main_table(viirs_tbl, opt.nenya_tbl_file)
    

def evaluate(opt_path, debug=False, clobber=False, 
             preproc:str='_std'):
    """
    This function is used to obtain the latents of the trained model
    for all of VIIRS

    Args:
        opt_path: (str) option file path.
        model_name: (str) model name 
        clobber: (bool, optional)
            If true, over-write any existing file
    """
    # Parse the model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    # Prep
    model_base, existing_files = latents_extraction.prep(opt)

    # Data files
    all_pp_files = ulmo_io.list_of_bucket_files('viirs', 'PreProc')
    pp_files = []
    for ifile in all_pp_files:
        if opt.eval_root in ifile:
            pp_files.append(f's3://viirs/{ifile}')


    for ifile in pp_files:
        print(f"Working on {ifile}")
        data_file = os.path.basename(ifile)

        # Setup
        latents_file = data_file.replace('_preproc', '_latents')
        if latents_file in existing_files and not clobber:
            print(f"Not clobbering {latents_file} in s3")
            continue

        s3_file = os.path.join(opt.s3_outdir, opt.latents_folder, latents_file) 

        # Download
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, ifile)
        else:
            print(f"Data file already downloaded: {data_file}")

        # Extract
        print("Extracting latents")
        latent_dict = latents_extraction.model_latents_extract(
            opt, data_file, model_base, debug=debug)
        # Save
        latents_hf = h5py.File(latents_file, 'w')
        for partition in latent_dict.keys():
            latents_hf.create_dataset(partition, data=latent_dict[partition])
        latents_hf.close()

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, s3_file)

        # Remove data file
        if not debug:
            os.remove(data_file)
            print(f'{data_file} removed')


def umap_me(opt_path:str, debug=False, local=True, 
            metric:str='DT'):
    """Run a UMAP analysis on all the VIIRS L2 data
    v5 model

    2 dimensions

    Args:
        model_name: (str) model name 
        ntrain (int, optional): Number of random latent vectors to use to train the UMAP model
        debug (bool, optional): For testing and debuggin 
        ndim (int, optional): Number of dimensions for the embedding
        metric (str, optional): Metric to use for UMAP
            TODO -- Use DT40
    """
    # Load up the options file
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    # Load v5 Table
    if local:
        tbl_file = os.path.join(os.getenv('OS_SST'),
                                'VIIRS', 'Tables', 
                                os.path.basename(opt.tbl_file))
    else:                            
        tbl_file = opt.tbl_file
    viirs_tbl = ulmo_io.load_main_table(tbl_file)

    # Add slope
    #viirs_tbl['min_slope'] = np.minimum(
    #    viirs_tbl.zonal_slope, viirs_tbl.merid_slope)

    # Base
    base1 = 'viirs_v1'

    if 'DT' in metric: 
        subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        if debug:
            subsets = ['DT5']
    elif metric == 'alpha':
        subsets = list(nenya_defs.umap_alpha.keys())
        if debug:
            subsets = ['a0']
    else:
        raise ValueError("Bad metric")

    # Loop me
    for subset in subsets:
        # Files
        outfile = os.path.join(
            os.getenv('OS_SST'), 
            f'VIIRS/Nenya/Tables/VIIRS_Nenya_{base1}_{subset}.parquet')
        if debug:
            umap_savefile = 'umap_test.pkl'
        else:
            umap_savefile = os.path.join(
                os.getenv('OS_SST'), 
                f'VIIRS/Nenya/UMAP/VIIRS_Nenya_{base1}_{subset}_UMAP.pkl')

        DT_cut = None 
        alpha_cut = None 
        if 'DT' in metric:
            # DT cut
            DT_cut = None if subset == 'DTall' else subset
        elif metric == 'alpha':
            alpha_cut = subset
        else:
            raise ValueError("Bad metric")

        #if debug:
        #    embed(header='86 of v5')

        # Run
        if os.path.isfile(umap_savefile):
            print(f"Skipping UMAP training as {umap_savefile} already exists")
            train_umap = False
        else:
            train_umap = True
        # Can't do both so quick check
        if DT_cut is not None and alpha_cut is not None:
            raise ValueError("Can't do both DT and alpha cuts")

        # Do it
        nenya_umap.umap_subset(viirs_tbl.copy(),
                             opt_path, 
                             outfile, 
                             local=local,
                             DT_cut=DT_cut, 
                             DT_key='DT',
                             alpha_cut=alpha_cut, 
                             debug=debug, 
                             train_umap=train_umap, 
                             umap_savefile=umap_savefile,
                             remove=False, CF=False)
        print(f"Done with {subset}")
    print("All done!")

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("func_flag", type=str, 
                        help="function to execute: train,evaluate,umap,umap_ndim3,sub2010,collect")
    parser.add_argument("--opt_path", type=str, 
                        default='opts_ssl_modis_v4.json',
                        help="Path to options file")
    parser.add_argument("--model", type=str, 
                        default='2010', help="Short name of the model used [2010,CF]")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('--local', default=False, action='store_true',
                        help='Local?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Redo?')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--ncpu", type=int, help="Number of CPUs")
    parser.add_argument("--years", type=str, help="Years to analyze")
    parser.add_argument("--cf", type=float, 
                        help="Clear fraction (e.g. 96)")
    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # Train the model
    if args.func_flag == 'train':
        print("Training Starts.")
        train(args.opt_path, debug=args.debug)
        print("Training Ends.")
        # python -u nenya_viirs.py train --opt_path opts_viirs_v1.json 

    # Prep Nenya Table
    if args.func_flag == 'prep_table':
        print("Prep Starts.")
        prep_nenya_table(args.opt_path, debug=args.debug)
        print("Prep Ends.")
        # python -u nenya_viirs.py prep_table --opt_path opts_viirs_v1.json 

    # Evaluate
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        evaluate(args.opt_path, debug=args.debug)
        print("Evaluation Ends.")
        # python -u nenya_viirs.py evaluate --opt_path opts_viirs_v1.json 

    # python ssl_modis_v4.py --func_flag umap --debug --local
    if args.func_flag == 'umap':
        umap_me(args.opt_path, debug=args.debug, local=args.local)
        # python -u nenya_viirs.py umap --opt_path opts_viirs_v1.json  --local

    '''
    # python ssl_modis_v4.py --func_flag revert_mask --debug
    if args.func_flag == 'revert_mask':
        revert_mask(debug=args.debug)

    # python ssl_modis_v4.py --func_flag preproc --debug
    if args.func_flag == 'preproc':
        modis_20s_preproc(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ulmo_evaluate --debug
    #  This comes before the slurp and cut
    if args.func_flag == 'ulmo_evaluate':
        modis_ulmo_evaluate(debug=args.debug)

    # python ssl_modis_v4.py --func_flag slurp_tables --debug
    if args.func_flag == 'slurp_tables':
        slurp_tables(debug=args.debug)

    # python ssl_modis_v4.py --func_flag cut_96 --debug
    #if args.func_flag == 'cut_96':
    #    cut_96(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ssl_evaluate --debug
    if args.func_flag == 'ssl_evaluate':
        main_ssl_evaluate(args.opt_path, debug=args.debug)
        
    # python ssl_modis_v4.py --func_flag DT40 --debug --local
    if args.func_flag == 'DT40':
        calc_dt40(args.opt_path, debug=args.debug, local=args.local,
                  redo=args.redo)

    # python ssl_modis_v4.py --func_flag umap --debug --local
    if args.func_flag == 'umap':
        ssl_v4_umap(args.opt_path, debug=args.debug, local=args.local)

    # Repeat UMAP analysis by DT using alpha instead
    # python ssl_modis_v4.py --func_flag alpha --debug --local
    if args.func_flag == 'alpha':
        ssl_v4_umap(args.opt_path, metric='alpha', debug=args.debug, local=args.local)
    '''

