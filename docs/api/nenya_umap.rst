.. _api_nenya_umap:

nenya_umap
=========

.. py:module:: nenya.nenya_umap

The ``nenya_umap`` module provides functionality for UMAP dimensionality reduction and analysis of latent spaces.

Functions
--------

.. py:function:: DT_interval(inp)

   Generate a DT (temperature difference) interval from the input.
   
   :param inp: DT central value and dDT, or None for all
   :type inp: tuple or None
   :return: Range of DT values as (min, max)
   :rtype: tuple

.. py:function:: load(model_name, DT=None, use_s3=False)

   Load a UMAP model.
   
   :param model_name: Model name ('LLC', 'LLC_local', 'CF', 'v4', 'v5', 'viirs_v1')
   :type model_name: str
   :param DT: DT value (K). Defaults to None.
   :type DT: float, optional
   :param use_s3: Whether to use S3 storage. Defaults to False.
   :type use_s3: bool, optional
   :return: Tuple of (UMAP model, table file path)
   :rtype: tuple
   :raises IOError: If model name is invalid or S3 is requested but not configured

.. py:function:: umap_subset(tbl, opt_path, outfile, DT_cut=None, alpha_cut=None, max_cloud_fraction=None, ntrain=200000, remove=True, DT_key='DT40', umap_savefile=None, train_umap=True, local=True, CF=False, debug=False)

   Run UMAP on a subset of the data. First 2 dimensions are written to the table.

   Run UMAP on a subset of the data. First 2 dimensions are written to the table.
   
   :param tbl: Data table
   :type tbl: pandas.DataFrame
   :param opt_path: Path to options file
   :type opt_path: str
   :param outfile: Output file path
   :type outfile: str
   :param DT_cut: DT cut to apply (e.g., 'DT2', 'DT4'). Defaults to None.
   :type DT_cut: str, optional
   :param alpha_cut: Alpha cut to apply (e.g., 'a1', 'a2'). Defaults to None.
   :type alpha_cut: str, optional
   :param max_cloud_fraction: Maximum cloud fraction to include. Defaults to None.
   :type max_cloud_fraction: float, optional
   :param ntrain: Number of samples to use for training UMAP. Defaults to 200000.
   :type ntrain: int, optional
   :param remove: Whether to remove temporary files. Defaults to True.
   :type remove: bool, optional
   :param DT_key: Key for DT values in the table. Defaults to 'DT40'.
   :type DT_key: str, optional
   :param umap_savefile: File to save the UMAP model. Defaults to None.
   :type umap_savefile: str, optional
   :param train_umap: Whether to train a new UMAP model. Defaults to True.
   :type train_umap: bool, optional
   :param local: Whether to use local files. Defaults to True.
   :type local: bool, optional
   :param CF: Whether to use cloud-free dataset. Defaults to False.
   :type CF: bool, optional
   :param debug: Whether to run in debug mode. Defaults to False.
   :type debug: bool, optional

.. py:function:: grid_umap(U0, U1, nxy=16, percent=[0.05, 99.95], verbose=False)

   Generate a grid on the UMAP domain.
   
   :param U0: First UMAP dimension coordinates
   :type U0: numpy.ndarray
   :param U1: Second UMAP dimension coordinates
   :type U1: numpy.ndarray
   :param nxy: Number of grid cells in each dimension. Defaults to 16.
   :type nxy: int, optional
   :param percent: Percentile range for grid boundaries. Defaults to [0.05, 99.95].
   :type percent: list, optional
   :param verbose: Whether to print details. Defaults to False.
   :type verbose: bool, optional
   :return: Dictionary containing grid information
   :rtype: dict

.. py:function:: cutouts_on_umap_grid(tbl, nxy, umap_keys, min_pts=1)

   Generate a list of cutouts uniformly distributed on the UMAP grid.
   
   :param tbl: Data table
   :type tbl: pandas.DataFrame
   :param nxy: Number of grid cells in each dimension
   :type nxy: int
   :param umap_keys: Tuple of column names for UMAP coordinates
   :type umap_keys: tuple
   :param min_pts: Minimum points required in each grid cell. Defaults to 1.
   :type min_pts: int, optional
   :return: Tuple of (filtered table, cutouts, umap_grid)
   :rtype: tuple

.. py:function:: regional_analysis(geo_region, tbl, nxy, umap_keys, min_counts=200)

   Analyze the distribution of a geographic region in UMAP space.
   
   :param geo_region: Name of the geographic region (defined in defs.py)
   :type geo_region: str
   :param tbl: Data table
   :type tbl: pandas.DataFrame
   :param nxy: Number of grid cells in each dimension
   :type nxy: int
   :param umap_keys: Tuple of column names for UMAP coordinates
   :type umap_keys: tuple
   :param min_counts: Minimum counts for normalization. Defaults to 200.
   :type min_counts: int, optional
   :return: Tuple of (counts, counts_geo, tbl, grid, xedges, yedges)
   :rtype: tuple