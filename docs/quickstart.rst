.. _quickstart:

Quickstart
=========

This guide will help you get up and running with Nenya quickly.

Basic Usage
----------

1. Loading a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nenya import io as nenya_io
   from nenya import analyze_image
   
   # Load model options and model file path
   opt, nenya_model_file = nenya_io.load_opt('v5')  # Available models: 'v4', 'v5', 'viirs_v1'

2. Analyzing a single image
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Create a sample image or load from file
   image = np.random.rand(64, 64)  # Or load your actual satellite image (64x64)
   
   # Extract latent representation and embed in UMAP space
   embedding, pp_img, table_file, DT, latents = analyze_image.umap_image('v5', image)
   
   print(f'UMAP coordinates: U0={embedding[0,0]:.3f}, U1={embedding[0,1]:.3f}')
   print(f'DT value: {DT:.2f}')

3. Working with UMAP
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nenya import nenya_umap
   
   # Load UMAP model
   umap_model, table_file = nenya_umap.load('v5')
   
   # Load table containing UMAP coordinates and metadata
   import pandas as pd
   umap_tbl = pd.read_parquet(table_file)
   
   # Plot UMAP (with matplotlib)
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 8))
   plt.scatter(umap_tbl.US0, umap_tbl.US1, s=1, alpha=0.5)
   plt.xlabel('U0')
   plt.ylabel('U1')
   plt.title('UMAP Embedding of Satellite Images')
   plt.show()

4. Interactive visualization portal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   from nenya.portal import OSSinglePortal
   from bokeh.server.server import Server
   
   # Path to the UMAP table
   table_file = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Nenya', 'Tables', 'MODIS_Nenya_96clear_v5_DT15.parquet')
   
   # Function to create a session
   def get_session(doc):
       sess = OSSinglePortal(table_file)
       return sess(doc)
   
   # Start Bokeh server
   server = Server({'/': get_session}, num_procs=1)
   server.start()
   print('Opening Bokeh application on http://localhost:5006/')
   
   server.io_loop.add_callback(server.show, "/")
   server.io_loop.start()

Training a New Model
------------------

.. code-block:: python

   from nenya.train import main as train_main
   
   # Path to options file
   opt_path = "path/to/opts_nenya_model.json"
   
   # Train model
   train_main(opt_path, debug=False)

Extracting Latents
----------------

.. code-block:: python

   from nenya.latents_extraction import model_latents_extract
   from nenya import io as nenya_io
   
   # Load options
   opt, model_path = nenya_io.load_opt('v5')
   
   # Data file path
   data_file = "path/to/preprocessed_data.h5"
   
   # Extract latents
   latent_dict = model_latents_extract(opt, data_file, model_path)
   
   # Access latents for valid and train sets
   valid_latents = latent_dict['valid']
   train_latents = latent_dict['train']

What's Next
----------

- Check out the :ref:`examples` section for more detailed usage examples
- Learn about the :ref:`concepts` behind Nenya
- Explore the :ref:`api-reference` for a comprehensive list of functions and classes
