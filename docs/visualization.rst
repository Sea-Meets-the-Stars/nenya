.. _visualization:

Interactive Visualization
======================

Nenya provides powerful interactive visualization tools through the `portal.py` module, which uses Bokeh to create a web-based interface for exploring satellite imagery and its latent space representations.

Starting the Portal
-----------------

To launch the interactive portal:

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

You can also provide an initial image:

.. code-block:: python

   from nenya.portal import Image, OSSinglePortal
   
   # Create an Image object
   input_Image = Image(image, Us=(0.5, 0.5), DT=2.5, lat=35.0, lon=-120.0)
   
   # Create portal with the image
   def get_session(doc):
       sess = OSSinglePortal(table_file, input_Image=input_Image)
       return sess(doc)

Portal User Interface
-------------------

The portal interface consists of several components:

1. **Primary Image View**: Displays the current selected image
2. **UMAP Plot**: Shows the 2D embedding of all images with the current image highlighted
3. **Image Gallery**: Displays a collection of similar images
4. **Matched Table**: Shows metadata for matched images
5. **Geographic View**: Displays the geographic locations of selected points

Primary Image Controls
~~~~~~~~~~~~~~~~~~~

- **DT, U0, U1 Text**: Displays metrics for the current image
- **PCB Low/High**: Controls the color mapping range
- **Use Input Image**: Resets to the original input image

UMAP Controls
~~~~~~~~~~~

- **Color by**: Select metric for coloring points (LL, DT, etc.)
- **Radius**: Set the matching radius for finding similar images
- **Alpha**: Adjust transparency of non-selected points
- **Set Img by View**: Set the primary image to the center of the current view

Gallery Controls
~~~~~~~~~~~~~

- **Inspect by**: Choose source for inspection (U = UMAP matches, geo = geographic)
- **Next/Previous Set**: Navigate through pages of matched images

Matched Table Controls
~~~~~~~~~~~~~~~~~~~

- **Set Img by Table**: Set the primary image to the selected table row

Geographic View
~~~~~~~~~~~~

- **Map**: Shows geographic distribution of matched points
- **Points**: Colored by the same metric as the UMAP plot

Working with the Portal Programmatically
--------------------------------------

Creating a Custom Portal
~~~~~~~~~~~~~~~~~~~~~

You can customize the portal by subclassing `OSSinglePortal`:

.. code-block:: python

   class CustomPortal(OSSinglePortal):
       def __init__(self, table_file, input_Image=None, init_Us=None):
           super().__init__(table_file, input_Image, init_Us)
           # Custom initialization
           
       def custom_method(self):
           # Custom functionality
           pass

Finding Similar Images
~~~~~~~~~~~~~~~~~~~

The portal uses the following methods to find similar images:

.. code-block:: python

   def set_matched(self, radius):
       """Find images within radius in UMAP space"""
       dist = (self.match_Us[0]-self.umap_tbl.US0.values)**2 + (
           self.match_Us[1]-self.umap_tbl.US1.values)**2
       # Matched
       matched = np.where(dist < radius**2)[0]
       if len(matched) == 0:
           self.match_idx = []
           return
       # Sort by distance
       srt = np.argsort(dist[matched])
       self.match_idx = matched[srt].tolist()

Handling Image Data
~~~~~~~~~~~~~~~~

Images are loaded from HDF5 files based on table metadata:

.. code-block:: python

   def load_images(self, tbl_idx):
       """Load images from disk based on table indices"""
       images, titles = [], []
       for kk in tbl_idx:
           if kk < 0:
               images.append(self.get_im_empty())
               titles.append(' ')
               continue
           #
           row = self.umap_tbl.iloc[kk]
           #
           ppfile = row.pp_file
           pp_idx = row.pp_idx
           # Grab
           base = os.path.basename(ppfile)
           key = 'valid' if row.ulmo_pp_type == 0 else 'train'
           img = self.file_dict[base][key][pp_idx, 0, ...]
           # Finish
           images.append(img)
           titles.append(str(kk))
       return images, titles

Callbacks and Event Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

The portal uses various callbacks to handle user interactions:

.. code-block:: python

   def register_callbacks(self):
       """Register callbacks for UI elements"""
       # Buttons
       self.prev_set.on_click(self.prev_set_callback)
       self.next_set.on_click(self.next_set_callback)
       self.Us_byview_set.on_click(self.Us_byview_callback)
       # ...
       
       # UMAP figure
       self.umap_figure.on_event(PanEnd, self.update_umap_filter_event())
       self.umap_figure.on_event(Reset, self.update_umap_filter_event(reset=True))
       
       # And many more...

Tips for Visualization
--------------------

1. **Memory Management**: The portal loads images on-demand to manage memory usage
2. **Performance**: For large datasets, use the decimation features to improve performance
3. **Customization**: Adjust color schemes, match radius, and alpha for better visualization
4. **Exploration**: Use the geographic view alongside UMAP to understand spatial patterns
5. **Selection**: Use the table view to inspect metadata for specific images

Advanced Portal Features
---------------------

Regional Analysis
~~~~~~~~~~~~~~~

The portal can visualize regional data distributions:

.. code-block:: python

   # Update the geo_source with filtered data
   def get_new_geo_view(self):
       """Update the geographic view based on current limits"""
       px_start, px_end, py_start, py_end = self.grab_geo_limits()
       viewed_objID = np.array(portal_utils.get_decimated_region_points(
           px_start, px_end, py_start, py_end, self.geo_source.data, 
           self.DECIMATE_NUMBER, IGNORE_TH=-9e9, id_key='obj_ID'))
       # Match and update
       # ...

Color Mapping
~~~~~~~~~~~

Control color mapping for visualizations:

.. code-block:: python

   def set_colormap(self, metric, metric_key):
       """Set color map range based on data distribution"""
       mx = np.nanmax(metric)
       mn = np.nanmin(metric)
       if mn == mx:
           high = mx + 1
           low = mn - 1
       else:
           high = mx + (mx - mn)*self.high_colormap_factor
           low = mn
           # Handle outliers
           nth = 100
           if len(metric)>nth:
               nmx = np.sort(metric)[-nth]
               if nmx*1.2 < mx:
                   high = nmx
       
       # Special case for log-likelihood
       if metric_key == 'LL':
           low = max(low, -1000.)
       
       self.umap_color_mapper.high = high
       self.umap_color_mapper.low = low
