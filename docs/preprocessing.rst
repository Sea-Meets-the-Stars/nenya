.. _preprocessing:

Data Preprocessing
================

Nenya implements several preprocessing steps to prepare satellite imagery data for model training and analysis.

Image Preprocessing Pipeline
--------------------------

The preprocessing pipeline includes the following steps:

1. **Demeaning**: Subtracting the mean value from the image
2. **Data Augmentation**: Applying transformations to create multiple views of each image
3. **Normalization**: Additional normalization steps before feeding to the network

Augmentation Classes
------------------

The main augmentation classes are defined in ``train_util.py``:

Demean
~~~~~~

Removes the mean from the image, which helps the model focus on relative temperature patterns rather than absolute values.

.. code-block:: python

   class Demean:
       def __call__(self, image):
           """Demean the input image by subtracting its mean"""
           image -= image.mean()
           return image

ThreeChannel
~~~~~~~~~~~

Converts single-channel images to three-channel format required by the ResNet architecture.

.. code-block:: python

   class ThreeChannel:
       def __call__(self, image):
           """Convert single-channel image to 3-channel format"""
           image = np.repeat(image, 3, axis=-1)
           return image

RandomRotate
~~~~~~~~~~~

Applies random rotation to images, enhancing rotation invariance in the model.

.. code-block:: python

   class RandomRotate:
       def __init__(self, verbose=False):
           self.verbose = verbose
           
       def __call__(self, image):
           """Apply random rotation to the image"""
           rang = np.float32(360*np.random.rand(1))
           return (skimage.transform.rotate(image, rang[0])).astype(np.float32)

RandomFlip
~~~~~~~~~

Randomly flips images horizontally and/or vertically.

.. code-block:: python

   class RandomFlip:
       def __init__(self, verbose=False):
           self.verbose = verbose
           
       def __call__(self, image):
           """Apply random flips to the image"""
           rflips = np.random.randint(2, size=2)
           if rflips[0] == 1:
               image = image[:, ::-1]  # Left/right flip
           if rflips[1] == 1:
               image = image[::-1, :]  # Up/down flip
           return image

JitterCrop
~~~~~~~~~

Crops images with random jitter for position invariance.

.. code-block:: python

   class JitterCrop:
       def __init__(self, crop_dim=32, rescale=2, jitter_lim=0, verbose=False):
           self.crop_dim = crop_dim
           self.offset = self.crop_dim//2
           self.jitter_lim = jitter_lim
           self.rescale = rescale
           self.verbose = verbose
           
       def __call__(self, image):
           """Crop with random jitter and optionally rescale"""
           # Implementation details...

TwoCropTransform
~~~~~~~~~~~~~~~

Creates two differently augmented views of the same image for contrastive learning.

.. code-block:: python

   class TwoCropTransform:
       """Create two transformations of the same image"""
       def __init__(self, transform):
           self.transform = transform

       def __call__(self, x):
           return [self.transform(x), self.transform(x)]

Creating Data Loaders
-------------------

Nenya provides functions to create data loaders with the appropriate transformations:

.. code-block:: python

   def nenya_loader(opt, valid=False):
       """Create a dataloader with augmentations based on options"""
       # Construct the augmentation list
       augment_list = []
       if opt.flip:
           augment_list.append(RandomFlip())
       if opt.rotate:
           augment_list.append(RandomRotate())
       if opt.random_jitter == 0:
           augment_list.append(JitterCrop())
       else:
           augment_list.append(JitterCrop(crop_dim=opt.random_jitter[0],
                                         jitter_lim=opt.random_jitter[1],
                                         rescale=0))
       if opt.demean:
           augment_list.append(Demean())

       # 3-channel augmentation
       augment_list.append(ThreeChannel())

       # Tensor conversion
       augment_list.append(transforms.ToTensor())
       
       # Create the data loader
       # ...

DT (Temperature Difference) Calculation
-------------------------------------

DT is a key metric calculated during preprocessing, representing the temperature gradient within an image:

.. code-block:: python

   def calc_DT(images, random_jitter, verbose=False):
       """Calculate DT (temperature difference) for given images
       
       DT is defined as T_90 - T_10, the difference between the 90th
       and 10th percentile temperatures in the center region of the image.
       """
       # Implementation details...
       
       # Calculate T90, T10
       T_90 = np.percentile(fields[..., xcen-dx:xcen+dx,
                                   ycen-dy:ycen+dy], 90., axis=(1,2))
       T_10 = np.percentile(fields[..., xcen-dx:xcen+dx,
                                   ycen-dy:ycen+dy], 10., axis=(1,2))
       DT = T_90 - T_10
       
       return DT

Data Format
---------

Nenya expects data in specific formats:

- Input images are typically 64x64 pixels (single channel)
- During preprocessing, images are converted to 3-channel format
- For training, images are organized in HDF5 files with 'train' and 'valid' datasets
- The preprocessing pipeline produces preprocessed files with '_preproc' suffix

Custom Dataset Classes
-------------------

Nenya provides custom dataset classes for various data types:

.. code-block:: python

   class NenyaDataset(Dataset):
       """Dataset used for training the Nenya model"""
       def __init__(self, data_path, transform, data_key='train'):
           self.data_path = data_path
           self.transform = transform
           self.data_key = data_key
       
       # Implementation details...

Tips for Preprocessing
-------------------

1. **Memory Management**: HDF5 files are read on-demand to manage memory usage for large datasets
2. **Batch Size**: Adjust batch size based on available GPU memory
3. **Workers**: Increase num_workers for faster data loading on systems with multiple CPUs
4. **Custom Augmentations**: Add additional augmentations by extending the augment_list

Preprocessing Command Line
------------------------

For batch preprocessing, Nenya provides command-line utilities (not shown in the uploaded code).
