.. _concepts:

Core Concepts
============

This page explains the key concepts and methodologies used in Nenya.

Self-Supervised Learning
----------------------

Nenya uses self-supervised learning (SSL) techniques to learn representations from satellite imagery without explicit labels. Specifically, it implements contrastive learning approaches:

SimCLR
~~~~~~

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) works by:

1. Taking an input image and creating two augmented views
2. Passing both through an encoder network to get feature representations
3. Applying a projection head to map representations to a space where contrastive loss is applied
4. Training the network to bring positive pairs (augmentations of the same image) closer while pushing negative pairs (augmentations of different images) apart

SupCon
~~~~~~

Supervised Contrastive Learning extends SimCLR by allowing the use of label information when available, though Nenya primarily uses the self-supervised approach.

Latent Space
-----------

The "latent space" refers to the compact representation of images learned by the model. In Nenya:

- Images are encoded into a feature vector (typically 128 or 512 dimensions)
- These vectors capture meaningful patterns and structures in the data
- Similar ocean features should have similar latent representations

UMAP Dimensionality Reduction
---------------------------

Uniform Manifold Approximation and Projection (UMAP) is used to reduce the high-dimensional latent space to 2D for visualization:

- UMAP preserves both local and global structure of the data
- The 2D coordinates (U0, U1) allow for visual exploration of the latent space
- Points close in the UMAP visualization represent similar ocean patterns

DT (Temperature Difference)
-------------------------

DT is a key metric in Nenya representing the temperature difference within an image:

- Calculated as the difference between the 90th and 10th percentile temperatures in the image
- Reflects the temperature gradient or contrast in the oceanic region
- Higher DT values typically indicate boundaries between water masses or frontal regions

Data Preprocessing
---------------

Before training or inference, images undergo several preprocessing steps:

- **Rotation**: Random rotations for data augmentation
- **Flipping**: Random horizontal and vertical flips
- **Jitter and Crop**: Random spatial jittering and cropping
- **Normalization**: Demeaning the image (subtracting the mean)

Data Organization
--------------

Nenya organizes data in several formats:

- **HDF5 files**: Store preprocessed images and extracted latent vectors
- **Parquet tables**: Store metadata and UMAP coordinates for efficient querying

Visualization Portal
-----------------

The interactive portal provides tools for exploring the latent space:

- **UMAP Plot**: Shows the 2D embedding of all images
- **Image Gallery**: Displays actual images corresponding to points in the UMAP space
- **Geographic View**: Shows the geographic distribution of selected points
- **Matching**: Finds similar images based on proximity in latent space

Models
-----

Nenya includes several pre-trained models:

- **v4**: MODIS imagery model (earlier version)
- **v5**: MODIS imagery model (improved version)
- **viirs_v1**: VIIRS imagery model
- **LLC**: MIT General Circulation Model imagery
- **CF**: Cloud-free specific model

Each model has been trained on specific satellite datasets and may be specialized for different oceanic phenomena.
