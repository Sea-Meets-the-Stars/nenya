""" Run Nenya on the N21 dataset """

import os
import json

from nenya.train import main as train_main

def main():

    # Train the model
    train_main("opts_nenya_viirs_test.json", debug=False)
    #train_main("opts_nenya_viirs.json", debug=False)

# Command line execution
if __name__ == '__main__':
    main()