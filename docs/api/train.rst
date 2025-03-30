.. _api_train:

train
====

.. py:module:: nenya.train

The ``train`` module provides functionality for training Nenya models using self-supervised contrastive learning.

Functions
--------

.. py:function:: main(opt_path, debug=False, save_file=None)

   Main function for training a Nenya model.
   
   :param opt_path: Path to the parameters JSON file
   :type opt_path: str
   :param debug: If True, run in debug mode with reduced epochs
   :type debug: bool, optional
   :param save_file: Path to save the trained model, if different from default
   :type save_file: str, optional
   
   This function:
   
   1. Loads parameters from the JSON file
   2. Sets up the model and criterion
   3. Configures the optimizer
   4. Trains the model for the specified number of epochs
   5. Periodically validates the model
   6. Saves model checkpoints and learning curves

Dependencies
-----------

This module depends on:

- ``nenya.io``: For loading and saving data
- ``nenya.train_util``: For model setup and training utilities
- ``nenya.params``: For parameter management
- ``nenya.util``: For optimization and model saving utilities

Training Process
--------------

The training process includes:

1. Loading parameters from a JSON file
2. Setting up the model and criterion
3. Setting up the optimizer
4. For each epoch:
   - Creating a data loader
   - Adjusting the learning rate
   - Training for one epoch
   - Recording losses
5. Optionally validating the model at specified intervals
6. Saving model checkpoints
7. Saving learning curves

Example Usage
-----------

.. code-block:: python

   from nenya.train import main as train_main
   
   # Train with parameters from a JSON file
   train_main("opts_nenya_modis_v5.json", debug=False)
   
   # Train in debug mode (reduced epochs)
   train_main("opts_nenya_modis_v5.json", debug=True)
   
   # Train and save to a custom location
   train_main("opts_nenya_modis_v5.json", save_file="/custom/path/model.pth")

Output Files
----------

After training, the following files are created:

1. Model checkpoints in ``{opt.model_folder}/ckpt_epoch_{epoch}.pth``
2. Final model in ``{opt.model_folder}/last.pth``
3. Learning curves in ``{opt.model_folder}/learning_curve/``
   - ``{opt.model_name}_losses_train.h5``: Training losses
   - ``{opt.model_name}_losses_valid.h5``: Validation losses

Model File Structure
-----------------

The saved model files have the following structure:

.. code-block:: python

   {
       'opt': opt,                 # Training options
       'model': model.state_dict(), # Model weights
       'optimizer': optimizer.state_dict(), # Optimizer state
       'epoch': epoch,             # Current epoch
   }

Learning Curve Files
-----------------

The learning curve HDF5 files contain:

- ``loss_train``: Array of training losses per epoch
- ``loss_step_train``: Array of per-step losses during training
- ``loss_avg_train``: Array of running average losses during training
- Similar arrays for validation losses

Related Modules
-------------

- :ref:`api_train_util`: Utilities for model training
- :ref:`api_params`: Parameter management
- :ref:`api_util`: Optimization and model saving utilities
