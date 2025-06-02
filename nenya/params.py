import os
import math

import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = -1.5  # change the value of learning_rate in params
    ```
    This module comes from:
    https://github.com/cs229-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    """

    def __init__(self, json_path):
        self.nenya_data = None
        self.data_folder = None
        self.lr_decay_epochs = None
        self.model_name = None
        self.cosine = None
        self.warmup_from = None
        self.warmup_to = None
        self.warmup_epochs = None
        self.model_folder = None
        self.latents_folder = None
        self.cuda_use = None
        self.valid_freq = None
        self.save_freq = None

        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=3)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

   
def option_preprocess(opt:Params):
    """
    Set up a number of preprocessing and more including
    output folders (e.g. model_folder, latents_folder)

    Object is modified in place.
    
    Args:
        opt: (Params) json used to store the training hyper-parameters
    """

    # check if dataset is path that passed required arguments
    if opt.nenya_data is True:
        assert opt.data_folder is not None, "Please provide data_folder in opt.json file." 

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './experimens/datasets/'
    

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.ssl_method, opt.ssl_model, opt.learning_rate,
               opt.weight_decay, opt.batch_size_train, opt.temp, opt.trial)

    # Cosine?
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size_train > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # Save folder
    opt.model_folder = os.path.join('models', opt.model_root,
                                    opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    # Latents folder
    opt.latents_folder = os.path.join('latents', opt.model_root,
                                    opt.model_name)

