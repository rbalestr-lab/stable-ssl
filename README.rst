Stable-SSL: the Self-Supervised Learning Library by Researchers for Researchers
===============================================================================

*You got a research idea? It shouldn't take you more than 10 minutes to start from scratch and get it running with the ability to produce high quality figures/tables from the results: that's the goal of stable-SSL.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: PytorchLightning, VISSL, Wandb, Hydra, Submitit.

``stable-SSL`` implements all the basic boilerplate code, including data loader, logging, checkpointing, optimization, etc. You only need to implement 3 methods to get started: your loss, your model, and your prediction (see `example <#own_trainer>`_ below). But if you want to customize more things, simply inherit the base ``BaseModel`` and override any method! This could include different metrics, different data samples, different training loops, etc.


.. image:: https://github.com/rbalestr-lab/stable-SSL/raw/main/docs/source/figures/logo.png
   :alt: ssl logo
   :width: 200px
   :align: right

.. contents:: Table of Contents
   :depth: 2


Why stable-SSL?
---------------

.. _why:

A quick search of ``AI libraries`` or ``Self Supervised Learning libraries`` will return hundreds of results. 99% will be independent project-centric libraries that can't be reused for general purpose AI research. The other 1% includes:

- Framework libraries such as PytorchLightning that focus on production needs.
- SSL libraries such as VISSL, FFCV-SSL, LightlySSL that are too rigid, often discontinued or not maintained, or commercial.
- Standalone libraries such as Wandb, submitit, Hydra that do not offer enough boilerplate for AI research.

Hence our goal is to fill that void.



Installation
------------

.. _installation:

The library is not yet available on PyPI. You can install it from the source code, as follows.

.. code-block:: bash

   pip install -e .

Or you can also run:

.. code-block:: bash

   pip install -U git+https://github.com/rbalestr-lab/stable-SSL



Minimal Documentation
---------------------

.. _minimal:


Implement your own `Trainer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _own_trainer:

At the very least, you need to implement three methods:

- ``initialize_modules``: this method initializes whatever model and parameters to use for training/inference
- ``forward``: that method that will be doing the prediction, e.g., for classification it will be p(y|x)
- ``compute_loss``: that method should return a scalar value used for backpropagation/training.


Library Design
~~~~~~~~~~~~~~

.. _design:

Stable-SSL provides all the boilerplate to quickly get started doing AI research, with a focus on Self Supervised Learning (SSL) albeit other applications can certainly build upon Stable-SSL. In short, we provide a ``BaseModel`` class that calls the following methods (in order):

.. code-block:: text

   1. INITIALIZATION PHASE:
     - seed_everything()
     - initialize_modules()
     - initialize_optimizer()
     - initialize_scheduler()
     - load_checkpoint()

   2. TRAIN/EVAL PHASE:
     - before_train_epoch()
     - for batch in train_loader:
       - before_train_step()
       - train_step(batch)
       - after_train_step()
     - after_train_epoch()

While the organization is related to the one provided by PytorchLightning, the goal here is to greatly reduce the codebase complexity without sacrificing performances. Think of PytorchLightning as industry driven (abstracting everything away) while Stable-SSL is academia driven (bringing everything in front of the user).

Examples
--------

.. _examples:

How to launch experiments
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _getting_started:

First build a confif file 

.. code-block:: python

   @hydra.main()
   def main(cfg):
      """Load the configuration and launch the run."""
      args = stable_ssl.get_args(cfg)  # Get the verified arguments
      model = getattr(stable_ssl, args.model.name)(args)  # Create model
      model()  # Call model


The file ``main.py`` to launch experiments is located in the ``runs/`` folder.

The default parameters are given in the ``sable_ssl/config.py`` file.
The parameters are structured in the following groups: data, model, hardware, log, optim.


Using default config files
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use default config files that are located in ``runs/configs``. To do so, simply specify the config file with the ``--config-name`` command as follows:

.. code-block:: bash

   python3 train.py --config-name=simclr_cifar10_sgd --config-path configs/

Classification case
~~~~~~~~~~~~~~~~~~~

- **How is the accuracy calculated?** The predictions are assumed to be the output of the forward method, then this is fed into a few metrics along with ``self.data[1]`` which is assumed to encode the labels.

Setting params in command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can modify/add parameters of the config file by adding ``++group.variable=value`` as follows:

.. code-block:: bash

   python3 main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
   # same but with SLURM
   python3 main.py --config-name=simclr_cifar10_sgd ++optim.epochs=4 ++optim.lr=1 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=1800 hydra.launcher.cpus_per_task=4 hydra.launcher.gpus_per_task=1 hydra.launcher.partition=gpu-he

**Remark**: If ``group.variable`` is already in the config file you can use ``group.variable=value`` and if it is not you can use ``+group.variable=value``. The ``++`` command handles both cases; thus we recommend using it.
