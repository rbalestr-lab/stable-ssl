
|Documentation| |Benchmark| |Test Status| |CircleCI| |Pytorch| |Ruff| |License| |WandB|


⚠️ This library is currently in a phase of active development. All features are subject to change without prior notice.


stable-ssl
==========

`stable-ssl`` provides all the boilerplate to quickly get started with AI research, focusing on Self-Supervised Learning (SSL), albeit other applications can certainly build upon ``stable-ssl``.
At its core, ``stable-ssl`` provides a `BaseTrainer <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer>`_ class that provides all the essential methods required to train and evaluate your model effectively. This class is intended to be subclassed for specific training needs (see these `trainers <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html>`_ as examples).

To make the process streamlined and efficient, we recommend using configuration files to define parameters and utilizing `Hydra <https://hydra.cc/>`_ to manage these configurations.

**General Idea.** ``stable-SSL`` provides a highly flexible framework with minimal hardcoded utilities. Modules in the pipeline can instantiate objects from various sources, including ``stable-SSL``, ``PyTorch``, ``TorchMetrics``, or even custom objects provided by the user. This allows you to seamlessly integrate your own components into the pipeline while leveraging the capabilities of ``stable-SSL``.

``stable-ssl`` implements all the basic boilerplate code, including job submission, data loading, optimization, evaluation, logging, monitoring, checkpointing, and requeuing. It offers users full flexibility to customize each part of the pipeline through a configuration file, enabling easy selection of network architectures, loss functions, evaluation metrics, data augmentations, and more.
These components can be sourced from ``stable-ssl`` itself, popular libraries like ``PyTorch``, or custom modules created by the user.


Why stable-ssl?
---------------

To start a run using the ``default_config.yaml`` configuration file located in the ``./configs/`` folder, use the following command:

.. code-block:: bash

   stable-ssl --config-path configs/ --config-name default_config

This command utilizes `Hydra <https://hydra.cc/>`_, making it compatible with multirun functionality and CLI overrides.
It is important to note that the multirun flag (``-m`` or ``--multirun``) is **mandatory** when using the Slurm launcher.


Structure your parameters
-------------------------

``stable-ssl`` uses ``Hydra`` (see the `Hydra documentation <https://hydra.cc/>`_) to manage input parameters via configuration files.
These parameters are grouped into the following categories (detailed in the `User Guide <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html>`_):

* **data**: defines the dataset, loading, and augmentation pipelines. Only the dataset called ``train`` is used for training. If there is no dataset named ``train``, the model runs in evaluation mode. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#data>`_.
* **module**: specifies the neural network modules. For instance: ``backbone``, ``projector``etc. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#module>`_.
* **optim**: contains optimization parameters, including ``epochs``, ``max_steps`` (per epoch), and ``optimizer`` / ``scheduler`` settings. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#optim>`_.
* **hardware**: specifies the hardware used, including the number of GPUs, CPUs, etc. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#hardware>`_.
* **logger**: configures model performance monitoring. APIs like `WandB <https://wandb.ai/home>`_ are supported. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#logger>`_.
* **loss** (optional): defines a loss function that can then be used in the ``compute_loss`` method of the trainer. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#loss>`_.

Structure your parameters
-------------------------

``stable-ssl`` uses ``Hydra`` (see the `Hydra documentation <https://hydra.cc/>`_) to manage input parameters via configuration files.
These parameters are grouped into the following categories (detailed in the `User Guide <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html>`_):

Data
----
Defines the dataset, loading, and augmentation pipelines. Only the dataset called ``train`` is used for training. If there is no dataset named ``train``, the model runs in evaluation mode. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#data>`_.

.. raw:: html

   <details>
   <summary>Example data YAML (click to reveal)</summary>

   ```yaml
   data:
     train:
       name: "CIFAR10"
       root: "/path/to/dataset"
       transform: "default_augmentation"
   ```
   </details>


Module
------
Specifies the neural network modules. For instance: ``backbone``, ``projector``, etc. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#module>`_.

.. raw:: html

   <details>
   <summary>Example module YAML (click to reveal)</summary>

   ```yaml
   module:
     backbone:
       name: "resnet50"
     projector:
       name: "mlp"
       hidden_dim: 2048
   ```
   </details>


Optim
-----
Contains optimization parameters, including ``epochs``, ``max_steps`` (per epoch), and ``optimizer`` / ``scheduler`` settings. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#optim>`_.

.. raw:: html

   <details>
   <summary>Example optim YAML (click to reveal)</summary>

   ```yaml
   optim:
     epochs: 100
     max_steps: null
     optimizer:
       name: "sgd"
       lr: 0.1
       momentum: 0.9
   ```
   </details>


Hardware
--------
Specifies the hardware used, including the number of GPUs, CPUs, etc. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#hardware>`_.

.. raw:: html

   <details>
   <summary>Example hardware YAML (click to reveal)</summary>

   ```yaml
   hardware:
     gpus: 1
     cpus: 8
     precision: 16
   ```
   </details>


Logger
------
Configures model performance monitoring. APIs like `WandB <https://wandb.ai/home>`_ are supported. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#logger>`_.

.. raw:: html

   <details>
   <summary>Example logger YAML (click to reveal)</summary>

   ```yaml
   logger:
     name: "wandb"
     project: "my_ssl_experiment"
     entity: "my_username"
   ```
   </details>


Loss (optional)
---------------
Defines a loss function that can then be used in the ``compute_loss`` method of the trainer. `Example <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#loss>`_.

.. raw:: html

   <details>
   <summary>Example loss YAML (click to reveal)</summary>

   ```yaml
   loss:
     name: "NTXEntLoss"
     temperature: 0.5
   ```
   </details>


Installation
------------

.. _installation:

The library is not yet available on PyPI. You can install it from the source code, as follows.

.. code-block:: bash

   pip install -e .

Or you can also run:

.. code-block:: bash

   pip install -U git+https://github.com/rbalestr-lab/stable-ssl


Minimal Documentation
---------------------

``stable-ssl`` provides all the boilerplate to quickly get started with AI research, focusing on Self-Supervised Learning (SSL), albeit other applications can certainly build upon ``stable-ssl``.
At its core, ``stable-ssl`` provides a `BaseTrainer <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer>`_ class that provides all the essential methods required to train and evaluate your model effectively. This class is intended to be subclassed for specific training needs (see these `trainers <https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html>`_ as examples).



.. |Documentation| image:: https://img.shields.io/badge/Documentation-blue.svg
    :target: https://rbalestr-lab.github.io/stable-ssl.github.io/dev/
.. |Benchmark| image:: https://img.shields.io/badge/Benchmarks-blue.svg
    :target: https://github.com/rbalestr-lab/stable-ssl/tree/main/benchmarks
.. |CircleCI| image:: https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-ssl/tree/main.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-ssl/tree/main
.. |Pytorch| image:: https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white
   :target: https://pytorch.org/get-started/locally/
.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |WandB| image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg
   :target: https://wandb.ai/site
.. |Test Status| image:: https://github.com/rbalestr-lab/stable-ssl/actions/workflows/testing.yml/badge.svg
