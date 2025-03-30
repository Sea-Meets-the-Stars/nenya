.. _api_analyze_image:

analyze_image
===========

.. py:module:: nenya.analyze_image

The ``analyze_image`` module provides functions for analyzing single images using Nenya models.

Functions
--------

.. py:function:: get_latents(img, model_file, opt)

   Get the Nenya latents for an input image.
   
   :param img: Input image (64,64) or (1,64,64)
   :type img: numpy.ndarray
   :param model_file: Full path to the Nenya model file (must be on local filesystem)
   :type model_file: str
   :param opt: Parameters for the Nenya model
   :type opt: nenya.params.Params
   :return: Tuple of (latents, pre-processed image)
   :rtype: tuple

.. py:function:: calc_DT(images, random_jitter, verbose=False)

   Calculate DT (temperature difference) for a given image or set of images.
   
   :param images: Input images. Analyzed shape is (N, 64, 64) but various input shapes are allowed.
   :type images: numpy.ndarray
   :param random_jitter: Range to crop, amount to randomly jitter
   :type random_jitter: list
   :param verbose: Whether to print progress. Defaults to False.
   :type verbose: bool, optional
   :return: DT value(s)
   :rtype: numpy.ndarray or float

.. py:function:: umap_image(nenya_model, img)

   UMAP embed an input image using a specified model.
   
   :param nenya_model: Nenya model name, e.g. 'v4', 'v5'
   :type nenya_model: str
   :param img: Input image
   :type img: numpy.ndarray
   :return: Tuple of (UMAP embedding, pre-processed image, table file path, DT value, latents)
   :rtype: tuple

Example Usage
-----------

.. code-block:: python

   from nenya import analyze_image
   from nenya import io as nenya_io
   import numpy as np
   
   # Create or load an image
   img = np.random.randn(64, 64)  # Example random image
   
   # Load model options
   opt, model_file = nenya_io.load_opt('v5')
   
   # Get latents
   latents, pp_img = analyze_image.get_latents(img, model_file, opt)
   
   # Calculate DT
   DT = analyze_image.calc_DT(img, opt.random_jitter)
   
   # UMAP embed the image
   embedding, pp_img, table_file, DT, latents = analyze_image.umap_image('v5', img)
   
   print(f"UMAP coordinates: U0={embedding[0,0]:.3f}, U1={embedding[0,1]:.3f}")
   print(f"DT value: {DT:.2f}")

Implementation Details
-------------------

The ``umap_image`` function combines several steps:

1. Loads the model options and model file
2. Extracts latent representations from the image
3. Calculates the DT value
4. Loads the appropriate UMAP model
5. Projects the latent vector into UMAP space

Image Preprocessing
----------------

Before extracting latents, images are preprocessed:

1. Reshaped to the expected format if necessary
2. Demeaned (mean subtracted)
3. Converted to a PyTorch tensor
4. Batched for model input

The ``calc_DT`` function:

1. Reshapes the input to a standard format
2. Calculates the 90th and 10th percentile temperatures in the center region
3. Returns the difference as the DT value

Related Modules
-------------

- :ref:`api_nenya_umap`: UMAP functionality for latent spaces
- :ref:`api_io`: Loading model options and files
- :ref:`api_latents_extraction`: Extracting latents from datasets
