.. _api_portal:

portal
=====

.. py:module:: nenya.portal

The ``portal`` module provides interactive visualization tools for exploring Nenya latent spaces and imagery.

Classes
------

.. py:class:: Image(image, Us, DT, lat=None, lon=None)

   Class representing an image and its metadata.
   
   :param image: The image data
   :type image: numpy.ndarray
   :param Us: UMAP coordinates (U0, U1)
   :type Us: tuple
   :param DT: Temperature difference value
   :type DT: float
   :param lat: Latitude (optional)
   :type lat: float, optional
   :param lon: Longitude (optional)
   :type lon: float, optional
   
   .. py:method:: copy()
   
      Create a deep copy of the Image object.
      
      :return: Copied Image object
      :rtype: Image

.. py:class:: OSSinglePortal(table_file, input_Image=None, init_Us=None)

   Main class for the interactive portal.
   
   :param table_file: Path to the UMAP table file
   :type table_file: str
   :param input_Image: Optional input image to start with
   :type input_Image: Image, optional
   :param init_Us: Initial UMAP coordinates if no image is provided
   :type init_Us: list, optional
   
   .. py:method:: __call__(doc)
   
      Add the portal layout to a Bokeh document.
      
      :param doc: Bokeh document
      :type doc: bokeh.document.Document
   
   .. py:method:: open_files()
   
      Open the HDF5 files containing images.
   
   .. py:method:: load_images(tbl_idx)
   
      Load images from disk based on table indices.
      
      :param tbl_idx: List of table indices to load
      :type tbl_idx: list
      :return: Tuple of (images, titles)
      :rtype: tuple
   
   .. py:method:: set_matched(radius)
   
      Find images within radius in UMAP space.
      
      :param radius: Radius for matching
      :type radius: float
   
   .. py:method:: find_closest_U(Us)
   
      Find the index of the closest point in UMAP space.
      
      :param Us: UMAP coordinates (U0, U1)
      :type Us: tuple
      :return: Index of the closest point
      :rtype: int
   
   .. py:method:: set_primary_by_objID(obj_ID)
   
      Set the primary image by object ID.
      
      :param obj_ID: Object ID in the table
      :type obj_ID: int
   
   .. py:method:: reset_from_primary()
   
      Reset the interface based on the primary image.
   
   .. py:method:: get_im_empty()
   
      Get an empty image of the correct size.
      
      :return: Empty image
      :rtype: numpy.ndarray

Example Usage
-----------

.. code-block:: python

   import os
   from nenya.portal import OSSinglePortal, Image
   from bokeh.server.server import Server
   import numpy as np
   
   # Path to UMAP table
   table_file = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Nenya', 'Tables', 'MODIS_Nenya_96clear_v5_DT15.parquet')
   
   # Optional: Create an Image object
   img = np.zeros((64, 64))  # Example image
   input_Image = Image(img, Us=(0.5, 0.5), DT=2.0, lat=35.0, lon=-120.0)
   
   # Create a session
   def get_session(doc):
       sess = OSSinglePortal(table_file, input_Image=input_Image)
       return sess(doc)
   
   # Start Bokeh server
   server = Server({'/': get_session}, num_procs=1)
   server.start()
   print('Opening Bokeh application on http://localhost:5006/')
   
   server.io_loop.add_callback(server.show, "/")
   server.io_loop.start()

Portal Components
---------------

The portal interface includes:

1. **Primary Image View**: 
   - Displays the current selected image
   - Shows DT, U0, U1 values
   - Controls for color mapping
   
2. **UMAP Plot**:
   - Shows the 2D embedding of all images
   - Highlights the current image and matches
   - Color-coded by selected metric (LL, DT, etc.)
   
3. **Gallery**:
   - Displays multiple images from matched set
   - Navigation controls for viewing more matches
   
4. **Data Table**:
   - Shows metadata for matched images
   - Allows selection for setting primary image
   
5. **Geographic View**:
   - Map showing geographic locations
   - Points colored by the same metric as UMAP

Technical Details
--------------

The portal uses:

- **Bokeh** for interactive visualization
- **HDF5** for efficient image access
- **Pandas** for table manipulation
- **NumPy** for numerical operations

Data Organization:

- Images are stored in HDF5 files referenced by the table
- The table contains metadata and UMAP coordinates
- Images are loaded on-demand to minimize memory usage

Customization:

- Color schemes can be adjusted
- Matching radius and transparency are configurable
- Metrics for coloring can be selected through the UI

Related Modules
-------------

- :ref:`api_nenya_umap`: UMAP analysis
- :ref:`api_analyze_image`: Single image analysis
