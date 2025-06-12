""" Run Nenya on the MNIST dataset """

import os

import nenya_utils


if 'OS_DATA' in os.environ:
    mnist_path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')
    preproc_file = os.path.join(mnist_path, 'PreProc', 'mnist_resampled.h5')
    latents_file = os.path.join(mnist_path, 'latents', 
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                        'mnist_resampled_latents.h5')

def evaluate():
    # Evaluate the model for MNIST
    latents_extraction.evaluate("opts_nenya_mnist.json", 
                os.path.join(mnist_path,'mnist_resampled.h5'), 
                local_model_path=mnist_path,
                debug=False, clobber=True)

def train():

    # Train the model
    train_main("opts_nenya_mnist.json", debug=False)

def main(task:str):
    if task == 'train':
        nenya_utils.train("opts_nenya_mnist.json", debug=False)
    elif task == 'evaluate':
        nenya_utils.evaluate()
    elif task == 'chk_latents':
        nenya_utils.chk_latents('MNIST', latents_file, preproc_file, 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = nenya_utils.return_task()
    main(task)