.. _installation:

Installation
===========

Prerequisites
------------

Nenya requires Python 3.7+ and the following main dependencies:

* PyTorch (1.7.0+)
* NumPy
* Pandas
* UMAP
* h5py
* Bokeh (for visualization)
* scikit-image
* tqdm

For working with geographic data:

* ulmo (internal dependency)

From PyPI
---------

.. note::
   Coming soon. The package is not yet available on PyPI.

.. code-block:: bash

   pip install nenya

From Source
----------

To install Nenya from source:

.. code-block:: bash

   git clone https://github.com/yourusername/nenya.git
   cd nenya
   pip install -e .

Environment Variables
--------------------

Nenya requires certain environment variables to be set for proper functioning:

* ``OS_SST``: Base directory for satellite data
* ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``: For accessing S3 data (if needed)

Example:

.. code-block:: bash

   export OS_SST=/path/to/sst/data

GPU Support
----------

For optimal performance, it's recommended to have a CUDA-compatible GPU. Nenya uses PyTorch's GPU acceleration when available.

To verify GPU detection:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0))

Docker Installation
-----------------

.. note::
   Docker configuration coming soon.

Development Installation
----------------------

For developers who want to contribute to Nenya:

.. code-block:: bash

   git clone https://github.com/yourusername/nenya.git
   cd nenya
   pip install -e ".[dev]"

This will install additional development dependencies like pytest, flake8, and sphinx for documentation.
