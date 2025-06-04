""" Run Nenya on the MNIST dataset """

import os

from nenya.train import main as train_main
from nenya import latents_extraction

mnist_path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')

def evaluate():
    # Evaluate the model for MNIST
    latents_extraction.evaluate("opts_nenya_mnist.json", 
                os.path.join(mnist_path,'mnist_resampled.h5'), 
                local_model_path=mnist_path,
                debug=False, clobber=True)

def train():

    # Train the model
    train_main("opts_nenya_mnist.json", debug=False)

# Command line execution
if __name__ == '__main__':
    # Train
    #train()

    # Evaluate
    evaluate()