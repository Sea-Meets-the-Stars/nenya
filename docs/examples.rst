.. _examples:

Examples
=======

This page provides complete examples of how to use Nenya for various tasks.

Analyzing an Individual Satellite Image
-------------------------------------

This example shows how to analyze a single image using a pre-trained Nenya model:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from nenya import analyze_image
   from nenya import io as nenya_io
   
   # Create a sample image (in practice, you would load from a file)
   # For example, a 64x64 image with a gradient
   x, y = np.meshgrid(np.linspace(-3, 3, 64), np.linspace(-3, 3, 64))
   r = np.sqrt(x**2 + y**2)
   image = np.exp(-0.1*r**2) + 0.1*np.random.randn(64, 64)
   
   # Analyze the image
   embedding, pp_img, table_file, DT, latents = analyze_image.umap_image('v5', image)
   
   # Print results
   print(f"UMAP coordinates: U0={embedding[0,0]:.3f}, U1={embedding[0,1]:.3f}")
   print(f"DT value: {DT:.2f}")
   print(f"Latent vector shape: {latents.shape}")
   
   # Plot the original and preprocessed images
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.imshow(image, cmap='turbo')
   plt.title('Original Image')
   plt.colorbar()
   
   plt.subplot(1, 2, 2)
   plt.imshow(pp_img[0, 0], cmap='turbo')
   plt.title('Preprocessed Image')
   plt.colorbar()
   
   plt.tight_layout()
   plt.show()

Exploring the UMAP Space
----------------------

This example shows how to visualize the UMAP space and explore patterns:

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from nenya import nenya_umap
   
   # Load UMAP model and table
   umap_model, table_file = nenya_umap.load('v5', DT=2.0)
   umap_tbl = pd.read_parquet(table_file)
   
   # Plot UMAP colored by DT
   plt.figure(figsize=(12, 10))
   scatter = plt.scatter(umap_tbl.US0, umap_tbl.US1, c=umap_tbl.DT40, 
                         s=1, alpha=0.5, cmap='plasma')
   plt.colorbar(scatter, label='DT (K)')
   plt.xlabel('U0')
   plt.ylabel('U1')
   plt.title('UMAP Embedding of Satellite Images')
   
   # Highlight geographic regions
   # Get points from a specific region (e.g., Pacific)
   pacific = (umap_tbl.lon > -170) & (umap_tbl.lon < -120) & \
             (umap_tbl.lat > -10) & (umap_tbl.lat < 10)
   
   plt.scatter(umap_tbl.US0[pacific], umap_tbl.US1[pacific], 
               s=5, color='red', alpha=0.7)
   
   plt.tight_layout()
   plt.show()
   
   # Create a geographic plot
   plt.figure(figsize=(12, 6))
   plt.scatter(umap_tbl.lon, umap_tbl.lat, c=umap_tbl.DT40, 
               s=1, alpha=0.5, cmap='plasma')
   plt.colorbar(label='DT (K)')
   plt.xlabel('Longitude')
   plt.ylabel('Latitude')
   plt.title('Geographic Distribution of Images')
   plt.grid(True)
   plt.tight_layout()
   plt.show()

Training a New Model
------------------

This example shows how to train a new Nenya model:

.. code-block:: python

   import os
   import json
   from nenya.train import main as train_main
   
   # Create a configuration file
   config = {
       "ssl_method": "SimCLR",
       "ssl_model": "resnet50",
       "learning_rate": 0.05,
       "weight_decay": 0.0001,
       "batch_size_train": 64,
       "batch_size_valid": 64,
       "temp": 0.07,
       "trial": 0,
       "cosine": True,
       "warm": True,
       "epochs": 200,
       "model_root": "my_model",
       "feat_dim": 128,
       "random_jitter": [5, 5],
       "images_file": "satellite_data_64x64.h5",
       "data_folder": "/path/to/data",
       "train_key": "train",
       "valid_key": "valid",
       "cuda_use": True,
       "valid_freq": 5,
       "save_freq": 10
   }
   
   # Save the configuration
   os.makedirs("configs", exist_ok=True)
   with open("configs/my_model_config.json", "w") as f:
       json.dump(config, f, indent=2)
   
   # Train the model
   train_main("configs/my_model_config.json", debug=False)

Extracting Latents from a Dataset
-------------------------------

This example shows how to extract latent vectors from a preprocessed dataset:

.. code-block:: python

   import os
   import h5py
   import numpy as np
   from nenya.latents_extraction import model_latents_extract
   from nenya import io as nenya_io
   
   # Load model options
   opt, model_path = nenya_io.load_opt('v5')
   
   # Extract latents from a preprocessed file
   data_file = "satellite_data_preproc.h5"
   latent_dict = model_latents_extract(opt, data_file, model_path)
   
   # Save latents to a new file
   with h5py.File("satellite_data_latents.h5", "w") as f:
       for key in latent_dict:
           f.create_dataset(key, data=latent_dict[key])
   
   # Print statistics
   for key in latent_dict:
       print(f"{key} latents shape: {latent_dict[key].shape}")
       print(f"{key} mean: {np.mean(latent_dict[key]):.4f}")
       print(f"{key} std: {np.std(latent_dict[key]):.4f}")

Creating a UMAP Model from Latents
--------------------------------

This example shows how to create a new UMAP model from extracted latents:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from nenya import nenya_umap
   
   # Create or load a table with metadata
   metadata = {
       'DT40': np.random.rand(1000) * 5,  # Sample DT values
       'lon': np.random.rand(1000) * 360 - 180,  # Longitudes
       'lat': np.random.rand(1000) * 180 - 90,   # Latitudes
       'pp_file': ['satellite_data_preproc.h5'] * 1000,  # Preprocessed file
       'pp_idx': np.arange(1000),  # Indices in the file
       'ulmo_pp_type': np.zeros(1000, dtype=int)  # 0 for valid set
   }
   tbl = pd.DataFrame(metadata)
   
   # Run UMAP on the data
   nenya_umap.umap_subset(
       tbl=tbl,
       opt_path='configs/my_model_config.json',
       outfile='my_model_umap.parquet',
       DT_cut='DT2',          # Filter by DT value
       ntrain=500,            # Number of samples to use for training
       umap_savefile='my_model_umap.pkl'  # Where to save the UMAP model
   )
   
   # Load the resulting table
   umap_tbl = pd.read_parquet('my_model_umap.parquet')
   
   # Print UMAP statistics
   print(f"U0 range: {umap_tbl.US0.min():.2f} to {umap_tbl.US0.max():.2f}")
   print(f"U1 range: {umap_tbl.US1.min():.2f} to {umap_tbl.US1.max():.2f}")

Interactive Visualization
-----------------------

This example shows how to create an interactive portal for data exploration:

.. code-block:: python

   import os
   import numpy as np
   from nenya.portal import OSSinglePortal, Image
   from bokeh.server.server import Server
   
   # Path to the UMAP table
   table_file = 'my_model_umap.parquet'
   
   # Create a sample image
   image = np.random.rand(64, 64)
   input_Image = Image(image, Us=(0.0, 0.0), DT=2.5, lat=0.0, lon=0.0)
   
   # Function to create a session
   def get_session(doc):
       sess = OSSinglePortal(table_file, input_Image=input_Image)
       return sess(doc)
   
   # Start Bokeh server
   server = Server({'/': get_session}, num_procs=1)
   server.start()
   print('Opening Bokeh application on http://localhost:5006/')
   
   server.io_loop.add_callback(server.show, "/")
   server.io_loop.start()
