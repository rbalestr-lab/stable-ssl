.. image:: https://github.com/rbalestr-lab/stable-SSL/raw/main/docs/source/figures/logo.jpg
   :alt: ssl logo
   :width: 200px
   :align: right

|Documentation| |Benchmark| |CircleCI| |Pytorch| |Black| |License| |WandB|


⚠️ This library is currently in a phase of active development. All features are subject to change without prior notice.


The Self-Supervised Learning Library by Researchers for Researchers
===================================================================

*Got a research idea? With stable-SSL, you can go from concept to execution in under 10 minutes. Start from scratch and quickly set up your pipeline, all while being able to generate high-quality figures and tables from your results. That's the goal of stable-SSL.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: ``PytorchLightning``, ``VISSL``, ``WandB``, ``Hydra``, ``Submitit``.

``stable-SSL`` implements all the basic boilerplate code, including job submission, data loading, optimization, evaluation, logging, monitoring, checkpointing, and requeuing. It offers users full flexibility to customize each part of the pipeline through a configuration file, enabling easy selection of network architectures, loss functions, evaluation metrics, data augmentations, and more.
These components can be sourced from ``stable-SSL`` itself, popular libraries like ``PyTorch``, or custom modules created by the user.

While the organization is similar to that of ``PyTorch Lightning``, the goal of ``stable-SSL`` is to significantly reduce codebase complexity without sacrificing performance. Think of ``PyTorch Lightning`` as industry-driven (abstracting everything away), whereas ``stable-SSL`` is academia-driven (offering complete visibility into all important aspects of the pipeline.).


Why stable-SSL?
---------------

.. _why:

A quick search of ``AI libraries`` or ``Self Supervised Learning libraries`` will return hundreds of results. 99% will be independent project-centric libraries that can't be reused for general purpose AI research. The other 1% includes:

- Framework libraries such as ``PytorchLightning`` that focus on production needs.
- SSL libraries such as ``VISSL``, ``FFCV-SSL``, ``LightlySSL`` that are discontinued, not maintained or too rigid.
- Standalone libraries such as ``Wandb``, ``submitit``, ``Hydra`` that do not offer enough boilerplate for AI research.

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

``stable-SSL`` provides all the boilerplate to quickly get started with AI research, focusing on Self-Supervised Learning (SSL), albeit other applications can certainly build upon ``stable-SSL``.
At its core, ``stable-SSL`` provides a `BaseTrainer <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer>`_ class that provides all the essential methods required to train and evaluate your model effectively. This class is intended to be subclassed for specific training needs (see these `trainers <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/trainers.html>`_ as examples).

``stable-SSL`` uses ``Hydra`` (see the `Hydra documentation <https://hydra.cc/>`_) to manage input parameters via configuration files. 
These parameters are grouped into the following categories (detailed in the `User Guide <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html>`_):

* **data**: Defines the dataset, loading, and augmentation pipelines. Only the dataset called ``train`` is used for training. If there is no dataset named ``train``, the model runs in evaluation mode. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#data>`_.
* **module**: Specifies the neural network modules, with a required ``backbone`` as the model's core. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#module>`_.
* **optim**: Contains optimization parameters, including ``epochs``, ``max_steps`` (per epoch), and ``optimizer`` / ``scheduler`` settings. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#optim>`_.
* **hardware**: Specifies the hardware used, including the number of GPUs, CPUs, etc. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#hardware>`_.
* **logger**: Configures model performance monitoring. APIs like `WandB <https://wandb.ai/home>`_ are supported. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#logger>`_.
* **loss** (optional): Defines a loss function that can then be used in the ``compute_loss`` method of the trainer. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#loss>`_.


To start a run using the ``default_config.yaml`` configuration file located in the ``./configs/`` folder, use the following command:

.. code-block:: bash

   stable-ssl --config-path configs/ --config-name default_config

This command utilizes `Hydra <https://hydra.cc/>`_, making it compatible with multirun functionality and CLI overrides.
It is important to note that the multirun flag (``-m`` or ``--multirun``) is **mandatory** when using the Slurm launcher.


.. Examples of Methods
.. ~~~~~~~~~~~~~~~~~~~

.. +----------------+--------------------------------------------+---------------------------------+
.. | **Method**     | **Trainer**                                | **Example Config**              |
.. +----------------+--------------------------------------------+---------------------------------+
.. | Barlow Twins   | `JointEmbeddingTrainer <jointembed_>`_     |                                 |
.. +----------------+--------------------------------------------+---------------------------------+
.. | BYOL           | `SelfDistillationTrainer <selfdistill_>`_  |                                 |
.. +----------------+--------------------------------------------+---------------------------------+
.. | DINO           | `DINOTrainer <dinotrainer_>`_              |                                 |
.. +----------------+--------------------------------------------+---------------------------------+
.. | MoCo           | `SelfDistillationTrainer <selfdistill_>`_  |                                 |
.. +----------------+--------------------------------------------+---------------------------------+
.. | SimCLR         | `JointEmbeddingTrainer <jointembed_>`_     | `link <exsimclr_>`_             |
.. +----------------+--------------------------------------------+---------------------------------+
.. | SimSiam        | `SelfDistillationTrainer <selfdistill_>`_  |                                 |
.. +----------------+--------------------------------------------+---------------------------------+
.. | VICReg         | `JointEmbeddingTrainer <jointembed_>`_     |                                 |
.. +----------------+--------------------------------------------+---------------------------------+


.. _exsimclr: https://github.com/huguesva/stable-SSL/tree/main/examples/config_examples/simclr_cifar10_full.yaml

.. _ntxent: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.NTXEntLoss.html#stable_ssl.losses.NTXEntLoss
.. _barlow: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.BarlowTwinsLoss.html#stable_ssl.losses.BarlowTwinsLoss
.. _negcosine: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.NegativeCosineSimilarity.html
.. _vicreg: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.VICRegLoss.html

.. _jointembed: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.JointEmbeddingTrainer.html
.. _selfdistill: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.SelfDistillationTrainer.html
.. _dinotrainer: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.DINOTrainer.html



.. |Documentation| image:: https://img.shields.io/badge/Documentation-blue.svg
    :target: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/
.. |Benchmark| image:: https://img.shields.io/badge/Benchmarks-blue.svg
    :target: https://github.com/rbalestr-lab/stable-SSL/tree/main/benchmarks
.. |CircleCI| image:: https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-SSL/tree/main.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-SSL/tree/main
.. |Pytorch| image:: https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white
   :target: https://pytorch.org/get-started/locally/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |WandB| image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg
   :target: https://wandb.ai/site
   