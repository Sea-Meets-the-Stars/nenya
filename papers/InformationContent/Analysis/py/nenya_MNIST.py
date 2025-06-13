""" Run Nenya on the MNIST dataset """

import os

from nenya import workflow



def main(task:str):
    if task == 'train':
        workflow.train(opts_file, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(opts_file, preproc_file, local_model_path=mnist_path,
                          latents_file=latents_file) 
    elif task == 'chk_latents':
        workflow.chk_latents('MNIST', latents_file, preproc_file, 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)