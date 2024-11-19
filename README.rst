.. image:: https://github.com/rbalestr-lab/stable-SSL/raw/main/docs/source/figures/logo.jpg
   :alt: ssl logo
   :width: 200px
   :align: right

|Documentation| |Benchmark| |CircleCI| |Pytorch| |Black| |License| |WandB|


⚠️ This library is currently in a phase of active development. All features are subject to change without prior notice.


The Self-Supervised Learning Library by Researchers for Researchers
===================================================================

*Have a research idea? With stable-SSL, you can go from concept to execution in under 10 minutes. Start from scratch and quickly set up your pipeline, all while being able to generate high-quality figures and tables from your results. That's the goal of stable-SSL.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: PytorchLightning, VISSL, Wandb, Hydra, Submitit.

``stable-SSL`` implements all the basic boilerplate code, including data loading, logging, checkpointing, optimization, etc. It offers users full flexibility to customize each part of the pipeline through a configuration file, enabling easy selection of network architectures, loss functions, evaluation metrics, data augmentations and more.
These components can be sourced from stable-SSL itself, popular libraries like PyTorch, or custom modules created by the user. This flexibility allows users to quickly get started with their research ideas.


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
     - before_fit_epoch()
     - for batch in train_loader:
       - before_fit_step()
       - fit_step(batch)
       - after_fit_step()
     - after_fit_epoch()

While the organization is related to the one provided by PytorchLightning, the goal here is to greatly reduce the codebase complexity without sacrificing performances. Think of PytorchLightning as industry driven (abstracting everything away) while Stable-SSL is academia driven (bringing everything in front of the user).


How to launch runs
------------------

.. _launch:

First build a config file with the parameters you want to use. The parameters should be structured in the following groups: data, model, hardware, log, optim.
See the :ref:`Configuration File Guide <config_guide>` for more details.

Then, create a Python script that will load the configuration and launch the run. Here is an example:

.. code-block:: python

   import stable_ssl
   import hydra

   @hydra.main()
   def main(cfg):
      """Load the configuration and launch the run."""
      args = stable_ssl.get_args(cfg)  # Get the verified arguments
      model = getattr(stable_ssl, args.model.name)(args)  # Create model
      model()  # Call model


To launch the run using the configuration file ``default_config.yaml`` located in the ``./configs/`` folder, use the following command:

.. code-block:: bash

   python3 train.py --config-name default_config --config-path configs/



.. |Documentation| image:: https://img.shields.io/badge/Documentation-blue.svg
    :target: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/
.. |Benchmark| image:: https://img.shields.io/badge/Benchmarks-blue.svg
    :target: https://github.com/rbalestr-lab/stable-SSL/tree/main/benchmarks
.. |CircleCI| image:: https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-SSL/tree/main.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-SSL/tree/main
.. |Pytorch| image:: https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white
    :target: https://pytorch.org/get-started/locally/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |WandB| image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg
   :target: https://wandb.ai/site