.. _api_models:

models
=====

.. py:module:: nenya.models

The ``models`` module contains neural network architectures used in Nenya for self-supervised learning.

Submodules
---------

nenya.models.resnet_big
~~~~~~~~~~~~~~~~~~~~~

.. py:module:: nenya.models.resnet_big

This module implements the ResNet backbone with projection head for contrastive learning.

Classes
------

.. py:class:: SupConResNet(name='resnet50', head='mlp', feat_dim=128)

   Base encoder network with projection head for contrastive learning.
   
   :param name: Name of the backbone architecture ('resnet18', 'resnet34', 'resnet50', etc.)
   :type name: str, optional
   :param head: Type of projection head ('linear', 'mlp')
   :type head: str, optional
   :param feat_dim: Dimension of the feature vector
   :type feat_dim: int, optional
   
   .. py:method:: forward(x)
   
      Forward pass through the network.
      
      :param x: Input tensor
      :type x: torch.Tensor
      :return: Feature vector
      :rtype: torch.Tensor
   
   .. py:method:: forward_feat(x)
   
      Extract features without the projection head.
      
      :param x: Input tensor
      :type x: torch.Tensor
      :return: Feature vector before projection
      :rtype: torch.Tensor

.. py:class:: SupCEResNet(name='resnet50', num_classes=10, head='mlp', feat_dim=128)

   Network for supervised contrastive learning with classification head.
   
   :param name: Name of the backbone architecture
   :type name: str, optional
   :param num_classes: Number of output classes
   :type num_classes: int, optional
   :param head: Type of projection head
   :type head: str, optional
   :param feat_dim: Dimension of the feature vector
   :type feat_dim: int, optional
   
   .. py:method:: forward(x)
   
      Forward pass through the network.
      
      :param x: Input tensor
      :type x: torch.Tensor
      :return: Classification logits
      :rtype: torch.Tensor
   
   .. py:method:: features(x)
   
      Extract features before the classification head.
      
      :param x: Input tensor
      :type x: torch.Tensor
      :return: Feature vector
      :rtype: torch.Tensor

Helper Functions
-------------

.. py:function:: nenya.models.resnet_big.get_resnet(name, pretrained=False)

   Get a ResNet model with a specific architecture.
   
   :param name: Name of the ResNet architecture ('resnet18', 'resnet34', 'resnet50', etc.)
   :type name: str
   :param pretrained: Whether to use pre-trained weights. Defaults to False.
   :type pretrained: bool, optional
   :return: ResNet model
   :rtype: torch.nn.Module
   :raises ValueError: If architecture name is not recognized

Architecture Details
-----------------

ResNet Backbone
~~~~~~~~~~~~~

The models use ResNet architectures with varying depths:

- **ResNet18**: 18 layers, ~11M parameters
- **ResNet34**: 34 layers, ~21M parameters
- **ResNet50**: 50 layers, ~23M parameters (default)
- **ResNet101**: 101 layers, ~42M parameters
- **ResNet152**: 152 layers, ~58M parameters

Projection Head
~~~~~~~~~~~~~

For contrastive learning, a projection head is added on top of the backbone:

- **Linear head**: Single linear layer
- **MLP head**: Two-layer MLP with ReLU activation (default)

The projection head maps the backbone features to a lower-dimensional space (typically 128 dimensions) where the contrastive loss is applied.

Usage Examples
-----------

Creating a model:

.. code-block:: python

   from nenya.models.resnet_big import SupConResNet
   
   # Create a ResNet50 model with 128-dimensional features
   model = SupConResNet(name='resnet50', feat_dim=128)
   
   # Forward pass
   import torch
   x = torch.randn(10, 3, 64, 64)  # Batch of 10 images
   features = model(x)  # Shape: [10, 128]

Model Input/Output
---------------

- **Input**: Images with shape [batch_size, 3, height, width]
- **Output**: Feature vectors with shape [batch_size, feat_dim]

For the contrastive learning setup:

- Input is augmented pairs of images: [2*batch_size, 3, height, width]
- Features are split and reshaped: [batch_size, 2, feat_dim]

Related Modules
-------------

- :ref:`api_train`: Training models
- :ref:`api_latents_extraction`: Extracting features from images
- :ref:`api_losses`: Contrastive loss functions
