.. _config_guide:

.. currentmodule:: stable_ssl

.. automodule:: stable_ssl
   :no-members:
   :no-inherited-members:


Configuration File Guide
========================

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


Weights and Biases Configuration (`wandb`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure Weights and Biases (Wandb) integration using the `wandb` keyword. The complete list of parameters for the `wandb` section is available below:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.WandbConfig


Data Configuration (`data`)   
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `data` keyword defines data loading, preprocessing and data augmentation settings. Here is an example of how to define the `data` section in your YAML file:

.. code-block:: yaml

   data:
      train_on: base
      base:
         name: CIFAR10
         batch_size: 32
         drop_last: True
         shuffle: True
         split: train
         num_workers: 10
         transforms:
            view1:
            - name: RandomResizedCrop
            kwargs:
               size: 32
               scale:
                  - 0.1
                  - 0.2
            - name: RandomHorizontalFlip
            - name: SpeckleNoise
            kwargs:
               severity: 2
            p: 0.5
            - name: GaussianBlur
            kwargs:
               kernel_size: 5
            p: 0.2
            view2:
            - name: RandomResizedCrop
            kwargs:
               size: 32
               scale:
                  - 0.1
                  - 0.2
            - name: RandomHorizontalFlip
      test_in:
         name: CIFAR10
         batch_size: 32
         drop_last: False
         split: train
         num_workers: 10
      test_out:
         name: CIFAR10
         batch_size: 32
         drop_last: False
         num_workers: 10
         split: test


The complete list of parameters for the `data` section can be found here:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.DataConfig


Model Configuration (`model`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `model` keyword is used to define the model settings, including the architecture of the backbone, objectives, and more. Below is a list of parameters shared across all models:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.BaseModelConfig

When defining a specific method, you can set method-specific parameters by creating a configuration class that inherits from `BaseModelConfig`. Examples of configurations for different methods in the library are provided below:

.. autosummary::   
   :toctree: gen_modules/
   :template: myclass_template.rst

   ssl_model.joint_embedding.SimCLRConfig
   ssl_model.joint_embedding.BarlowTwinsConfig
   ssl_model.joint_embedding.VICRegConfig
   ssl_model.joint_embedding.WMSEConfig
