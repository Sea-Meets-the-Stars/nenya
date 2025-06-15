""" Run Nenya on the MNIST dataset """

import os

from nenya import workflow

import info_defs


def main(task:str):
    dataset = 'MNIST'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], debug=False)
    elif task == 'evaluate':
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'], clobber=True) 
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    elif task == 'eigenimages':
        workflow.find_eigenmodes(pdict['opts_file'], pdict['pca_file'], 
                                 (1,28,28), f'{dataset}_eigenimages.npz',
                                 local_model_path=pdict['path'],
                                 num_iterations=10000)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)