.. _user_guide:

.. currentmodule:: stable_ssl

.. automodule:: stable_ssl
   :no-members:
   :no-inherited-members:


User Guide
==========

This guide explains how to construct a configuration file to launch a run with ``stable_ssl``. The configuration file is written in YAML format, and various sections map to different configuration classes corresponding to optimization, hardware, log, data and model settings.


Optimization Configuration (`optim`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `optim` keyword is used to define the optimization settings for your model. Here's an example of how to define the `optim` section in your YAML file:

.. code-block:: yaml

   optim:
     lr: 0.001
     batch_size: 256
     epochs: 500


The complete list of parameters for the `optim` section, including their descriptions and default values, is provided below:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.OptimConfig


Hardware Configuration (`hardware`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the `hardware` keyword to configure hardware-related settings such as the number of workers or GPU usage. The complete list of parameters for the `hardware` section is available below:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.HardwareConfig


Log Configuration (`log`)
~~~~~~~~~~~~~~~~~~~~~~~~~

The `log` keyword configures the logging settings for your run. The complete list of parameters for the `log` section is provided here:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.LogConfig


Data Configuration (`data`)   
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `data` keyword defines data loading, preprocessing and data augmentation settings. Here is an example of how to define the `data` section in your YAML file:

.. code-block:: yaml

   _num_classes: 10
   _num_samples: 50000
   base:
   _target_: torch.utils.data.DataLoader
   batch_size: 256
   drop_last: True
   shuffle: True
   num_workers: ${trainer.hardware.cpus_per_task}
   dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ~/data
      train: True
      transform:
         _target_: stable_ssl.data.MultiViewSampler
         transforms:
         - _target_: torchvision.transforms.v2.Compose
            transforms:
               - _target_: torchvision.transforms.v2.RandomResizedCrop
               size: 32
               scale:
                  - 0.2
                  - 1.0
               - _target_: torchvision.transforms.v2.RandomHorizontalFlip
               p: 0.5
               - _target_: torchvision.transforms.v2.ToImage
               - _target_: torchvision.transforms.v2.ToDtype
               dtype: 
                  _target_: stable_ssl.utils.str_to_dtype
                  _args_: [float32]
               scale: True
         - ${trainer.data.base.dataset.transform.transforms.0}
   test_out:
   _target_: torch.utils.data.DataLoader
   batch_size: 256
   num_workers: ${trainer.hardware.cpus_per_task}
   dataset:
      _target_: torchvision.datasets.CIFAR10
      train: False
      root: ~/data
      transform:
         _target_: torchvision.transforms.v2.Compose
         transforms:
         - _target_: torchvision.transforms.v2.ToImage
         - _target_: torchvision.transforms.v2.ToDtype
            dtype: 
               _target_: stable_ssl.utils.str_to_dtype
               _args_: [float32]
            scale: True


Model Configuration (`model`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `model` keyword is used to define the model settings, including the architecture of the backbone, objectives, and more. Below is a list of parameters shared across all models:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.BaseModelConfig

When defining a specific method, you can set method-specific parameters by creating a configuration class that inherits from `BaseModelConfig`. Examples of configurations for different methods in the library are provided below:


