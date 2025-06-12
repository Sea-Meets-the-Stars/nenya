""" Run Nenya on the ImageNet dataset """

from importlib import reload
import os
import shutil


import nenya_utils


if 'OS_SST' in os.environ:
    modis_path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
    preproc_file = os.path.join(modis_path, 'PreProc', 'train_MODIS_2021_128x128.h5')
    latents_file = os.path.join(modis_path, 'latents', 'MODIS_2021',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                        'train_MODIS_2021_128x128_latents.h5')

from IPython import embed

def main(task:str):
    if task == 'train':
        train()
    elif task == 'evaluate':
        evaluate()
    elif task == 'chk_latents':
        chk_latents(100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = nenya_utils.return_task()
    main(task)