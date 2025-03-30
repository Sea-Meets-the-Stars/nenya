.. _api_latents_extraction:

latents_extraction
================

.. py:module:: nenya.latents_extraction

The ``latents_extraction`` module provides functionality for extracting latent representations from preprocessed images.

Classes
------

.. py:class:: HDF5RGBDataset(file_path, partition, allowed_indices=None)

   A PyTorch dataset for HDF5 data used in latent extraction.
   
   :param file_path: Path to the HDF5 file
   :type file_path: str
   :param partition: Dataset name in the HDF5 file (e.g., 'train', 'valid')
   :type partition: str
   :param allowed_indices: Set of image indices to include (defaults to all)
   :type allowed_indices: numpy.ndarray, optional
   
   .. py:method:: __len__()
   
      Return the number of samples in the dataset.
      
      :return: Number of samples
      :rtype: int
      
   .. py:method:: __getitem__(index)
   
      Get a sample from the dataset.
      
      :param index: Index of the sample
      :type index: int
      :return: Tuple of (data, metadata)
      :rtype: tuple

Functions
--------

.. py:function:: main(opt_path, pp_files, clobber=False, debug=False)

   Main function for batch latent extraction.
   
   :param opt_path: Path to options file
   :type opt_path: str
   :param pp_files: List of preprocessed file paths
   :type pp_files: list
   :param clobber: Whether to overwrite existing files. Defaults to False.
   :type clobber: bool, optional
   :param debug: Whether to run in debug mode. Defaults to False.
   :type debug: bool, optional

.. py:function:: build_loader(data_file, dataset, batch_size=1, num_workers=1, allowed_indices=None)

   Create a data loader for latent extraction.
   
   :param data_file: Path to the data file
   :type data_file: str
   :param dataset: Dataset name in the file (e.g., 'train', 'valid')
   :type dataset: str
   :param batch_size: Batch size for data loading. Defaults to 1.
   :type batch_size: int, optional
   :param num_workers: Number of worker processes. Defaults to 1.
   :type num_workers: int, optional
   :param allowed_indices: Set of image indices to include. Defaults to None (all).
   :type allowed_indices: numpy.ndarray, optional
   :return: Tuple of (dataset, data loader)
   :rtype: tuple

.. py:function:: calc_latent(model, image_tensor, using_gpu)

   Calculate latent representations for an image tensor.
   
   :param model: Nenya model
   :type model: torch.nn.Module
   :param image_tensor: Image tensor
   :type image_tensor: torch.Tensor
   :param using_gpu: Whether to use GPU
   :type using_gpu: bool
   :return: Latent vectors as numpy array
   :rtype: numpy.ndarray

.. py:function:: prep(opt)

   Prepare the environment for latent extraction.
   
   :param opt: Model options
   :type opt: nenya.params.Params
   :return: Tuple of (model base name, list of existing latent files)
   :rtype: tuple

.. py:function:: model_latents_extract(opt, data_file, model_path, remove_module=True, in_loader=None, partitions=('train', 'valid'), allowed_indices=None, debug=False)

   Extract latents from a data file using a model.
   
   :param opt: Model options
   :type opt: nenya.params.Params
   :param data_file: Path to the data file
   :type data_file: str
   :param model_path: Path to the model file
   :type model_path: str
   :param remove_module: Whether to remove 'module.' prefix from keys. Defaults to True.
   :type remove_module: bool, optional
   :param in_loader: Optional pre-configured data loader. Defaults to None.
   :type in_loader: torch.utils.data.DataLoader, optional
   :param partitions: Dataset partitions to process. Defaults to ('train', 'valid').
   :type partitions: tuple, optional
   :param allowed_indices: Set of image indices to include. Defaults to None (all).
   :type allowed_indices: numpy.ndarray, optional
   :param debug: Whether to run in debug mode. Defaults to False.
   :type debug: bool, optional
   :return: Dictionary of latent vectors for each partition
   :rtype: dict

Example Usage
-----------

.. code-block:: python

   from nenya.latents_extraction import model_latents_extract, main
   from nenya import io as nenya_io
   
   # Extract latents for specific files
   pp_files = [
       's3://bucket/PreProc/data_file1_preproc.h5',
       's3://bucket/PreProc/data_file2_preproc.h5'
   ]
   
   # Batch extraction
   main("path/to/opts.json", pp_files, clobber=False)
   
   # Individual extraction
   opt, model_path = nenya_io.load_opt('v5')
   latent_dict = model_latents_extract(opt, "data_file_preproc.h5", model_path)
   
   # Access latents
   valid_latents = latent_dict['valid']
   train_latents = latent_dict['train']

Implementation Details
-------------------

The latent extraction process:

1. Loads the model and its weights
2. Creates data loaders for each partition in the data file
3. Passes batches of images through the model
4. Collects the latent vectors
5. Returns a dictionary with latent vectors for each partition

When using the ``main`` function, the process also includes:

1. Downloading files from S3 if necessary
2. Checking for existing latent files to avoid duplicating work
3. Saving extracted latents to HDF5 files
4. Uploading results to S3
5. Cleaning up temporary files

Related Modules
-------------

- :ref:`api_train_util`: Model setup and utilities
- :ref:`api_io`: I/O utilities for models and data
- :ref:`api_nenya_umap`: UMAP analysis of extracted latents
