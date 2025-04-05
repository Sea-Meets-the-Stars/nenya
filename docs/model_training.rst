.. _model_training:

Model Training
============

Nenya implements self-supervised contrastive learning for training models on satellite imagery. This page describes the training process, model architecture, and configuration options.

Training Process
--------------

The main training loop is implemented in ``train.py``:

.. code-block:: python

   from nenya.train import main as train_main
   
   # Train a model using options from a JSON file
   train_main("path/to/opts_file.json", debug=False)

The training function handles:

1. Loading and augmentation options
2. Building the model and criterion (loss function)
3. Setting up the optimizer
4. Training for the specified number of epochs
5. Validating the model periodically
6. Saving checkpoints and final model

Model Architecture
---------------

The default model architecture is based on ResNet with a customized 
projection head for contrastive learning.  The model is set with
the `set_model()` method in ``train_util.py``:

.. code-block:: python

   from nenya.models.resnet_big import SupConResNet
   
   # Create a model with a specific backbone and feature dimension
   model = SupConResNet(name='resnet50', feat_dim=128)

Nenya supports several ResNet variants ('resnet18', 'resnet34', 'resnet50', etc.) 
which can be specified in the options file.

Loss Function
-----------

Nenya uses a contrastive loss function implemented in ``losses.py``:

.. code-block:: python

   from nenya.losses import SupConLoss
   
   # Create a loss function with a specific temperature
   criterion = SupConLoss(temperature=0.07)

The contrastive loss encourages representations of different augmentations of the same image to be similar, while pushing representations of different images apart.

Option Configuration
-----------------

Training options are specified in a JSON file and loaded using the ``Params`` class:

.. code-block:: python

   from nenya import params
   
   # Load options from a JSON file
   opt = params.Params("path/to/opts_file.json")
   
   # Preprocess options (set derived values)
   params.option_preprocess(opt)

Key training options include:

.. code-block:: javascript

   {
     "ssl_method": "SimCLR",      // Training method (SimCLR or SupCon)
     "ssl_model": "resnet50",     // Backbone model
     "learning_rate": 0.05,       // Initial learning rate
     "batch_size_train": 64,      // Batch size for training
     "batch_size_valid": 64,      // Batch size for validation
     "epochs": 200,               // Number of epochs
     "feat_dim": 128,             // Feature dimension size
     "temp": 0.07,                // Temperature parameter for loss
     "weight_decay": 1e-4,        // Weight decay for optimizer
     "momentum": 0.9,             // Momentum for optimizer
     "cosine": true,              // Use cosine learning rate schedule
     "random_cropjitter": [40, 5],// Crop and Jitter (random) parameters for augmentation
     "rotate": true,              // Apply random rotation
     "flip": true,                // Apply random horizontal/vertical flip
     "demean": true,              // Apply mean normalization after crop
     "gauss_noise": 0.,           // Apply Gaussian noise; 0 = None
     "model_root": "models/v5",   // Root directory for model output
     "train_key": "train",        // Dataset key for training
     "valid_key": "valid",        // Dataset key for validation
     "save_freq": 10,             // Save checkpoint every N epochs
     "valid_freq": 5              // Validate every N epochs
   }

Data Loaders
-----------

Training and validation data loaders are created using the ``nenya_loader`` function:

.. code-block:: python

   from nenya.train_util import nenya_loader
   
   # Create a training data loader
   train_loader = nenya_loader(opt, valid=False)
   
   # Create a validation data loader
   valid_loader = nenya_loader(opt, valid=True)

These loaders apply the appropriate transformations and augmentations to the input images.

Training Loop
-----------

The core training loop is implemented in ``train_model``:

.. code-block:: python

   from nenya.train_util import train_model
   
   # Train for one epoch
   loss, losses_step, losses_avg = train_model(
       train_loader, model, criterion, optimizer, epoch, opt, 
       cuda_use=opt.cuda_use)

For each batch, the function:

1. Loads images and applies augmentations
2. Forwards the augmented views through the model
3. Calculates the contrastive loss
4. Updates the model parameters through backpropagation

Learning Rate Schedule
-------------------

Nenya supports learning rate warmup and decay:

.. code-block:: python

   from nenya.util import adjust_learning_rate, warmup_learning_rate
   
   # Adjust learning rate according to epoch
   adjust_learning_rate(opt, optimizer, epoch)
   
   # Apply warmup to the learning rate within an epoch
   warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

Model Saving
----------

Models are saved periodically during training and at the end:

.. code-block:: python

   from nenya.util import save_model
   
   # Save model checkpoint
   save_file = os.path.join(opt.model_folder, f'ckpt_epoch_{epoch}.pth')
   save_model(model, optimizer, opt, epoch, save_file)

Monitoring Training
-----------------

Training progress is monitored using the ``AverageMeter`` class:

.. code-block:: python

   from nenya.util import AverageMeter
   
   # Create meters for tracking statistics
   batch_time = AverageMeter()
   data_time = AverageMeter()
   losses = AverageMeter()
   
   # Update meter with new values
   losses.update(loss.item(), bsz)

Learning curves (loss over time) are saved to HDF5 files for later analysis:

.. code-block:: python

   with h5py.File(losses_file_train, 'w') as f:
       f.create_dataset('loss_train', data=np.array(loss_train))
       f.create_dataset('loss_step_train', data=np.array(loss_step_train))
       f.create_dataset('loss_avg_train', data=np.array(loss_avg_train))

Multi-GPU Training
---------------

Nenya supports multi-GPU training through PyTorch's DataParallel:

.. code-block:: python

   if torch.cuda.is_available() and cuda_use:
       if torch.cuda.device_count() > 1:
           model.encoder = torch.nn.DataParallel(model.encoder)
       model = model.cuda()
       criterion = criterion.cuda()
       cudnn.benchmark = True

Training Tips
-----------

1. **Batch Size**: Larger batch sizes generally work better for contrastive learning. If GPU memory is limited, consider using gradient accumulation.
2. **Temperature**: The temperature parameter in the loss function controls the concentration of the distribution. Lower values (e.g., 0.07) typically work well.
3. **Learning Rate**: A cosine learning rate schedule with warmup often leads to better results.
4. **Augmentations**: Strong augmentations are crucial for contrastive learning. Experiment with different combinations of rotation, jitter, and flips.
5. **Feature Dimension**: A higher feature dimension (e.g., 128 or 256) generally captures more information but requires more GPU memory.
