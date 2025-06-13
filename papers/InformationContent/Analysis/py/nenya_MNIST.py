""" Run Nenya on the MNIST dataset """

import os

from nenya import workflow


if 'OS_DATA' in os.environ:
    mnist_path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')
    preproc_file = os.path.join(mnist_path, 'PreProc', 'mnist_resampled.h5')
    latents_file = os.path.join(mnist_path, 'latents', 
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                        'mnist_resampled_latents.h5')
opts_file = 'opts_nenya_mnist.json'

def main(task:str):
    if task == 'train':
        workflow.train(opts_file, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(opts_file, preproc_file, local_model_path=mnist_path) 
    elif task == 'chk_latents':
        workflow.chk_latents('MNIST', latents_file, preproc_file, 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)