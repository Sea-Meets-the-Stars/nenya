""" Run Nenya on the N21 dataset """

import os
import shutil
import json

from nenya.train import main as train_main
from nenya import latents_extraction

modis_path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')

def evaluate():
    # Copy in the model
    #model_file = os.path.join(modis_path, 'models', 'MODIS_2021',
    #                    'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
    #                    'last.pth')
    #shutil.copy(model_file, './')

    # Evaluate the model for MODIS native
    latents_extraction.evaluate("opts_nenya_modis.json", 
                os.path.join(modis_path, 'PreProc', 'train_MODIS_2021_128x128.h5'),
                local_model_path=modis_path,
                use_gpu=False,
                debug=False, clobber=True)

def train():
    # Train the model
    #train_main("opts_nenya_modis.json", debug=False)
    train_main("opts_nenya_modis.json", debug=False, load_epoch=23)

# Command line execution
if __name__ == '__main__':

    # Train
    #train()

    # Evaluate
    evaluate()