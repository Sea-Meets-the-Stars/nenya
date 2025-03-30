.. Nenya documentation master file

Welcome to Nenya's documentation!
================================

Nenya is a machine learning framework for analyzing ocean satellite imagery, particularly sea surface temperature data from MODIS and VIIRS sensors. It uses self-supervised learning techniques and dimensionality reduction (UMAP) to analyze and visualize patterns in ocean imagery.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
-----------

* Self-supervised learning with contrastive approaches (SimCLR/SupCon)
* Extraction of meaningful latent space representations from satellite images
* UMAP dimensionality reduction for visualization
* Interactive web portal for data exploration
* Utilities for working with MODIS and VIIRS data
* Tools for regional ocean analysis

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts
   preprocessing
   model_training
   latent_extraction
   umap_analysis
   visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/params
   api/train
   api/nenya_umap
   api/analyze_image
   api/io
   api/latents_extraction
   api/models
   api/portal

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
