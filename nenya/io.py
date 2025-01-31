""" I/O routines for SSL analysis """
import os
from importlib import resources

from nenya import params

# THIS HACK IS NECESSARY FOR OLDER EXISTING, SERIALIZED MODELS
#  i.e. DON'T REMOVE IT!
Params = params.Params

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
    