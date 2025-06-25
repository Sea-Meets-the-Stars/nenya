""" Run Nenya on the LLC nonoise dataset """

import os
import numpy as np
from nenya import workflow
import info_defs

import h5py

from IPython import embed

def main(task:str):
    dataset = 'LLC_SST_noise'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], load_epoch=49, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'], debug=False)
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    else:
        raise ValueError(f"Unknown task: {task}")

def reset_learning(trial:int):
    # Load the learning curve files
    dataset = 'LLC_SST_noise'
    pdict = info_defs.grab_paths(dataset)
    path = pdict['path']
    valid_file = os.path.join(path, 'models', 'LLC_noise', 
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                              'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_valid.h5')
    train_file = os.path.join(path, 'models', 'LLC_noise', 
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                              'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_train.h5')

    # Open
    for ifile, key in zip([train_file, valid_file], ['train', 'valid']):
        print(f"Opening {ifile}")
        f = h5py.File(ifile, 'r')
        loss = np.array(f[f'loss_{key}'][:]).tolist()
        loss_step= np.array(f[f'loss_step_{key}'][:]).tolist()
        loss_avg= np.array(f[f'loss_avg_{key}'][:]).tolist()
        # Cut down
        loss = loss[:trial]
        f.close()
        # Write back
        f = h5py.File(ifile, 'w')
        f.create_dataset(f'loss_{key}', data=np.array(loss))
        f.create_dataset(f'loss_step_{key}', data=np.array(loss_step))
        f.create_dataset(f'loss_avg_{key}', data=np.array(loss_avg))
        f.close()
        print(f"Wrote {ifile}")


# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)

    #reset_learning(49)