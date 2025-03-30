.. _api_io:

io
==

.. py:module:: nenya.io

The ``io`` module provides I/O utilities for Nenya models and data.

Functions
--------

.. py:function:: load_opt(nenya_model)

   Load the options for a Nenya model.
   
   :param nenya_model: Name of the model (e.g., 'LLC', 'CF', 'v4', 'v5', 'viirs_v1')
   :type nenya_model: str
   :return: Tuple of (Nenya options, model file path)
   :rtype: tuple
   :raises IOError: If the model name is invalid

Example Usage
-----------

.. code-block:: python

   from nenya import io as nenya_io
   
   # Load options for a specific model
   opt, model_file = nenya_io.load_opt('v5')
   
   # Access option parameters
   print(f"Random jitter: {opt.random_jitter}")
   print(f"Model file: {model_file}")

Models
-----

The module supports several pre-defined models:

- ``'LLC'``: Large-scale model trained on LLC data
- ``'LLC_local'``: Local version of LLC model
- ``'CF'``: Cloud-free model
- ``'v4'``: MODIS version 4 model
- ``'v5'``: MODIS version 5 model
- ``'viirs_v1'``: VIIRS version 1 model

For each model, the function loads:

1. The appropriate options file containing model hyperparameters
2. The path to the model weights file

Implementation Details
-------------------

The function determines the paths to options files and model files based on the input model name:

.. code-block:: python

   if nenya_model == 'LLC' or nenya_model == 'LLC_local':
       # LLC model paths
       nenya_model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
       opt_path = os.path.join(resources.files('ulmo'), 'runs',
                            'Nenya', 'LLC', 'experiments', 
                            'llc_modis_2012', 'opts.json')
   elif nenya_model == 'CF': 
       # Cloud-free model paths
       opt_path = os.path.join(resources.files('ulmo'), 'runs',
           'Nenya', 'MODIS', 'v2', 'experiments',
           'modis_model_v2', 'opts_cloud_free.json')
   # ... other models

The options are loaded using the ``params.Params`` class and preprocessed with ``params.option_preprocess``.

Related Modules
-------------

- :ref:`api_params`: Parameter handling
- :ref:`api_analyze_image`: Image analysis using loaded models
- :ref:`api_latents_extraction`: Latent extraction using loaded models
