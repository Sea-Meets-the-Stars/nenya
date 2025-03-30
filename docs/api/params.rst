.. _api_params:

params
=====

.. py:module:: nenya.params

The ``params`` module handles configuration parameters for Nenya models and processing.

Classes
------

.. py:class:: Params(json_path)

   A class for loading and managing hyperparameters from a JSON file.
   
   :param json_path: Path to the JSON configuration file
   :type json_path: str
   
   .. py:method:: save(json_path)
   
      Save parameters to a JSON file.
      
      :param json_path: Path where the JSON file will be saved
      :type json_path: str
      
   .. py:method:: update(json_path)
   
      Update parameters from a JSON file.
      
      :param json_path: Path to the JSON configuration file
      :type json_path: str
      
   .. py:property:: dict
   
      Returns the parameters as a dictionary.
      
      :return: Dictionary of parameters
      :rtype: dict

Functions
--------

.. py:function:: option_preprocess(opt)

   Process options and set derived values.
   
   This function sets up output folders (e.g., model_folder, latents_folder) and
   processes other options. The object is modified in place.
   
   :param opt: Options object to preprocess
   :type opt: Params
   
Example Usage
-----------

.. code-block:: python

   from nenya import params
   
   # Load parameters from a JSON file
   opt = params.Params("opts_nenya_modis_v5.json")
   
   # Preprocess options
   params.option_preprocess(opt)
   
   # Access parameters
   learning_rate = opt.learning_rate
   batch_size = opt.batch_size_train
   
   # Save parameters to a new file
   opt.save("opts_new.json")

Parameter Structure
-----------------

Example parameter file structure:

.. code-block:: json

   {
     "ssl_method": "SimCLR",
     "ssl_model": "resnet50",
     "learning_rate": 0.05,
     "weight_decay": 0.0001,
     "batch_size_train": 64,
     "batch_size_valid": 64,
     "temp": 0.07,
     "trial": 0,
     "cosine": true,
     "warm": true,
     "epochs": 200,
     "model_root": "v5",
     "feat_dim": 128,
     "random_jitter": [5, 5],
     "images_file": "MODIS_2012_96clear_64x64.h5",
     "s3_outdir": "s3://bucket/path",
     "data_folder": "/path/to/data",
     "train_key": "train",
     "valid_key": "valid",
     "cuda_use": true,
     "valid_freq": 5,
     "save_freq": 10
   }
