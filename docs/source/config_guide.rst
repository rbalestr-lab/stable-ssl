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

The `data` keyword defines data loading, preprocessing and data augmentation settings. The complete list of parameters for the `data` section can be found here:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.DataConfig


Base Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   config.BaseModelConfig


.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   ssl_model.joint_embedding.SimCLRConfig
   ssl_model.joint_embedding.BarlowTwinsConfig
   ssl_model.joint_embedding.VICRegConfig
   ssl_model.joint_embedding.WMSEConfig
