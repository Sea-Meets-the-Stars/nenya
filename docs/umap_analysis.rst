.. _umap_analysis:

UMAP Analysis
===========

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique used in Nenya to visualize high-dimensional latent spaces in 2D. This enables exploration of patterns and relationships in satellite imagery data.

Loading UMAP Models
-----------------

To load a pre-trained UMAP model:

.. code-block:: python

   from nenya import nenya_umap
   
   # Load UMAP model for a specific Nenya model
   umap_model, table_file = nenya_umap.load('v5', DT=2.5)
   
   # Available models: 'LLC', 'LLC_local', 'CF', 'v4', 'v5', 'viirs_v1'
   # Optional DT parameter filters by temperature difference

The ``load`` function returns:

1. A trained UMAP model that can project new data
2. Path to a table file with pre-computed UMAP coordinates for the dataset

Creating a UMAP Model
-------------------

To create a new UMAP model from latent vectors:

.. code-block:: python

   import pandas as pd
   from nenya import nenya_umap
   
   # Load table with metadata
   tbl = pd.read_parquet('path/to/table.parquet')
   
   # Run UMAP on the data
   nenya_umap.umap_subset(
       tbl=tbl,
       opt_path='path/to/opts.json',
       outfile='output_table.parquet',
       DT_cut='DT2',          # Filter by DT value
       ntrain=200000,         # Number of samples to use for training
       umap_savefile='umap_model.pkl'  # Where to save the UMAP model
   )

This function:

1. Filters the data based on specified criteria (DT, alpha, etc.)
2. Loads latent vectors for the selected data
3. Trains a UMAP model on a random subset
4. Projects all data to the 2D UMAP space
5. Saves the results to a new table file

UMAP DT Filtering
---------------

UMAP models can be filtered by DT (temperature difference) to focus on specific oceanic features:

.. code-block:: python

   # DT intervals are defined in nenya.defs
   umap_DT = {
       'DT0': (0.25, 0.25),   # DT around 0.25K (±0.25)
       'DT1': (0.75, 0.25),   # DT around 0.75K (±0.25)
       'DT15': (1.25, 0.25),  # DT around 1.25K (±0.25)
       'DT2': (2.0, 0.5),     # DT around 2K (±0.5)
       'DT4': (3.25, 0.75),   # DT around 3.25K (±0.75)
       'DT5': (4.0, -1),      # DT >= 4K
       'all': None            # No DT filtering
   }

To apply a DT filter:

.. code-block:: python

   # For DT around 2K (±0.5)
   umap_model, table_file = nenya_umap.load('v5', DT=2.0)

Working with UMAP Coordinates
---------------------------

The resulting table contains UMAP coordinates as columns 'US0' and 'US1':

.. code-block:: python

   import pandas as pd
   
   # Load table with UMAP coordinates
   umap_tbl = pd.read_parquet(table_file)
   
   # Access UMAP coordinates
   u0 = umap_tbl.US0.values
   u1 = umap_tbl.US1.values
   
   # Plot UMAP coordinates
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 8))
   plt.scatter(u0, u1, s=1, alpha=0.5)
   plt.xlabel('U0')
   plt.ylabel('U1')
   plt.title('UMAP Embedding')
   plt.show()

Creating a UMAP Grid
------------------

To create a regular grid in UMAP space for analysis:

.. code-block:: python

   # Create a grid with 16x16 cells
   umap_grid = nenya_umap.grid_umap(
       umap_tbl.US0.values,
       umap_tbl.US1.values,
       nxy=16,
       percent=[0.05, 99.95]  # Percentile range to use for boundaries
   )
   
   # The grid contains:
   # - xmin, xmax, ymin, ymax: Boundaries
   # - xval, yval: Grid edge coordinates
   # - dxv, dyv: Cell dimensions

Selecting Cutouts with UMAP
-------------------------

To select representative cutouts across UMAP space:

.. code-block:: python

   # Select cutouts uniformly distributed in UMAP space
   filtered_tbl, cutouts, umap_grid = nenya_umap.cutouts_on_umap_grid(
       tbl=umap_tbl,
       nxy=16,
       umap_keys=('US0', 'US1'),
       min_pts=1  # Minimum points required in each grid cell
   )
   
   # cutouts is a list of rows from tbl, one for each grid cell (or None if empty)

Regional Analysis with UMAP
-------------------------

To analyze geographic regions in UMAP space:

.. code-block:: python

   # Analyze a specific geographic region
   counts, counts_geo, tbl, grid, xedges, yedges = nenya_umap.regional_analysis(
       geo_region='eqpacific',  # Name of region defined in defs.py
       tbl=umap_tbl,
       nxy=16,
       umap_keys=('US0', 'US1'),
       min_counts=200
   )
   
   # counts: Histogram of all points
   # counts_geo: Histogram of points in the region
   # grid: Grid information
   # xedges, yedges: Histogram bin edges

Geographic regions are defined in ``defs.py``:

.. code-block:: python

   geo_regions = {
       'coastalcali': {'lons': [-128, -118], 'lats': [32, 40]},
       'eqpacific': {'lons': [-140, -90], 'lats': [-5, 5]},
       'eqindian': {'lons': [60, 90], 'lats': [-5, 5]},
       # And others...
   }

Embedding New Images
------------------

To embed a new image in an existing UMAP space:

.. code-block:: python

   from nenya import analyze_image
   
   # Embed a single image in UMAP space
   embedding, pp_img, table_file, DT, latents = analyze_image.umap_image('v5', image)
   
   # embedding contains the UMAP coordinates (U0, U1) for the image

This function:

1. Loads the appropriate Nenya model
2. Extracts latent vectors from the image
3. Calculates DT (temperature difference)
4. Projects the latent vector to UMAP space
5. Returns the UMAP coordinates and other information

Visualizing UMAP Embeddings
-------------------------

For interactive visualization, use the portal functionality described in :ref:`visualization`.

Tips for UMAP Analysis
-------------------

1. **Training Size**: UMAP works well with a subset of the data (e.g., 200,000 samples)
2. **Filtering**: Consider filtering by DT or other criteria to focus on specific phenomena
3. **Normalization**: Normalize latent vectors before UMAP if not already done
4. **Parameters**: Experiment with UMAP parameters (n_neighbors, min_dist) if needed
5. **Geographic Analysis**: Compare UMAP patterns with geographic distributions
