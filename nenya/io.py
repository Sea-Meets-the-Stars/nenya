""" I/O routines for SSL analysis """
import os
from importlib import resources
from io import BytesIO

import pandas
import torch

from wrangler import s3_io as wr_io

from nenya import params
from nenya.train_util import set_model

# THIS HACK IS NECESSARY FOR OLDER EXISTING, SERIALIZED MODELS
#  i.e. DON'T REMOVE IT!
Params = params.Params

from IPython import embed

def load_opt(nenya_model:str):
    """ Load the SSL model options

    Args:
        nenya_model (str): name of the model
            e.g. 'LLC', 'CF', 'v4', 'v5'

    Raises:
        IOError: _description_

    Returns:
        tuple: Nenya options, model file (str)
    """
    #embed(header='load_opt 24')
    # Prep
    nenya_model_file = None
    if nenya_model == 'LLC' or nenya_model == 'LLC_local':
        nenya_model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_path = os.path.join(resources.files('ulmo'), 'runs',
                                'Nenya', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
    elif nenya_model == 'CF': 
        opt_path = os.path.join(resources.files('ulmo'), 'runs',
            'Nenya', 'MODIS', 'v2', 'experiments',
            'modis_model_v2', 'opts_cloud_free.json')
    elif nenya_model == 'v4':  
        opt_path = os.path.join(resources.files('ulmo'), 'runs',
            'Nenya', 'MODIS', 'v4', 'opts_nenya_modis_v4.json')
    elif nenya_model == 'v5': 
        opt_path = os.path.join(resources.files('ulmo'), 'runs',
            'Nenya', 'MODIS', 'v5', 'opts_nenya_modis_v5.json')
        nenya_model_file = os.path.join(os.getenv('OS_SST'),
                                  'MODIS_L2', 'Nenya', 'models', 
                                  'v4_last.pth')  # Only the UMAP was retrained (for now)
    elif nenya_model == 'viirs_v1': 
        opt_path  = os.path.join(
            resources.files('nenya'), '../', 'runs', 'viirs', 'opts_viirs_v1.json')
        nenya_model_file = os.path.join(os.getenv('OS_SST'),
                                  'VIIRS', 'Nenya', 'models', 
                                  'nenya_viirs_v1_last.pth')
    else:
        raise IOError("Bad model!!")

    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    if nenya_model_file is None:
        nenya_model_file = os.path.join(opt.s3_outdir, 
                                  opt.model_folder, 'last.pth')

    # Return
    return opt, nenya_model_file
    
def load_model_name(opt_path:str, local_model_path:str=None,
                    base_model_name:str=None):

    # Parse the model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    # Prep

    # Grab the model?
    if local_model_path is not None: 
        model_name = os.path.join(local_model_path,
            opt.model_folder, base_model_name)
    elif not os.path.isfile(base_model_name):
        print(f"Grabbing model: {model_file}")
        # Download the model from S3
        s3_model_file = os.path.join(opt.s3_outdir,
            opt.model_folder, base_model_name)
        ulmo_io.download_file_from_s3(model_name, s3_model_file)
    else:
        print(f"Model was already downloaded: {model_name}")

    return opt, model_name

def load_model(model_path:str, opt, using_gpu:bool,
               remove_module:bool=True, weights_only:bool=False): 
    # Specify the model
    model, _ = set_model(opt, cuda_use=using_gpu)

    # Load model
    if not using_gpu:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=weights_only)
    else:
        model_dict = torch.load(model_path, weights_only=weights_only)

    # Remove module?
    if remove_module:
        new_dict = {}
        for key in model_dict['model'].keys():
            new_dict[key.replace('module.','')] = model_dict['model'][key]
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(model_dict['model'])
    print("Model loaded")

    return model



def load_main_table(tbl_file:str, verbose=True):
    """Load the table of cutouts 

    Args:
        tbl_file (str): Path to table of cutouts. Local or s3
        verbose (bool, optional): [description]. Defaults to True.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: table of cutouts
    """
    _, file_extension = os.path.splitext(tbl_file)

    # s3?
    if tbl_file[0:5] == 's3://':
        inp = wr_io.load_to_bytes(tbl_file)
    else:
        inp = tbl_file
        
    # Allow for various formats
    if file_extension == '.csv':
        main_table = pandas.read_csv(inp, index_col=0)
        # Set time
        if 'datetime' in main_table.keys():
            main_table.datetime = pandas.to_datetime(main_table.datetime)
    elif file_extension == '.feather':
        # Allow for s3
        main_table = pandas.read_feather(inp)
    elif file_extension == '.parquet':
        # Allow for s3
        main_table = pandas.read_parquet(inp)
    else:
        raise IOError("Bad table extension: ")

    # Deal with masked int columns
    for key in ['gradb_Npos', 'FS_Npos', 'UID', 'pp_type']:
        if key in main_table.keys():
            main_table[key] = pandas.array(main_table[key].values, dtype='Int64')
    # Report
    if verbose:
        print("Read main table: {}".format(tbl_file))

    # Decorate
    if 'DT' not in main_table.keys() and 'T90' in main_table.keys():
        main_table['DT'] = main_table.T90 - main_table.T10
        
    return main_table

def write_main_table(main_table:pandas.DataFrame, outfile:str, to_s3=True):
    """Write Main table for ULMO analysis
    Format is determined from the outfile extension.
        Options are ".csv", ".feather", ".parquet"

    Args:
        main_table (pandas.DataFrame): Main table for ULMO analysis
        outfile (str): Output filename.  Its extension sets the format
        to_s3 (bool, optional): If True, write to s3

    Raises:
        IOError: [description]
    """
    _, file_extension = os.path.splitext(outfile)
    if file_extension == '.csv':
        main_table.to_csv(outfile, date_format='%Y-%m-%d %H:%M:%S')
    elif file_extension == '.feather':
        bytes_ = BytesIO()
        main_table.to_feather(path=bytes_)
        if to_s3:
            wr_io.write_bytes_to_s3(bytes_, outfile)
        else:
            wr_io.write_bytes_to_local(bytes_, outfile)
    elif file_extension == '.parquet':
        bytes_ = BytesIO()
        main_table.to_parquet(path=bytes_)
        if to_s3:
            wr_io.write_bytes_to_s3(bytes_, outfile)
        else:
            wr_io.write_bytes_to_local(bytes_, outfile)
    else:
        raise IOError("Not ready for this")
    print("Wrote Analysis Table: {}".format(outfile))


def losses_filenames(opt):

    losses_file_train = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_train.h5')
    losses_file_valid = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_valid.h5')
    # Return
    return losses_file_train, losses_file_valid 