.. image:: https://github.com/rbalestr-lab/stable-SSL/raw/main/docs/source/figures/logo.jpg
   :alt: ssl logo
   :width: 200px
   :align: right

|Documentation| |Benchmark| |CircleCI| |Pytorch| |Black| |License| |WandB|


⚠️ This library is currently in a phase of active development. All features are subject to change without prior notice.


The Self-Supervised Learning Library by Researchers for Researchers
===================================================================

*Have a research idea? With stable-SSL, you can go from concept to execution in under 10 minutes. Start from scratch and quickly set up your pipeline, all while being able to generate high-quality figures and tables from your results. That's the goal of stable-SSL.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: ``PytorchLightning``, ``VISSL``, ``WandB``, ``Hydra``, ``Submitit``.

``stable-SSL`` implements all the basic boilerplate code, including data loading, logging, checkpointing and optimization. It offers users full flexibility to customize each part of the pipeline through a configuration file, enabling easy selection of network architectures, loss functions, evaluation metrics, data augmentations and more.
These components can be sourced from ``stable-SSL`` itself, popular libraries like ``PyTorch``, or custom modules created by the user.


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

Library Design
~~~~~~~~~~~~~~

.. _design:

``stable-SSL`` provides all the boilerplate to quickly get started with AI research, focusing on Self-Supervised Learning (SSL), albeit other applications can certainly build upon ``stable-SSL``.
At its core, ``stable-SSL`` provides a `BaseTrainer <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer>`_ class that provides all the essential methods required to train and evaluate your model effectively. See the `User Guide <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html>`_ for guidance on configuring parameters to suit your use case.

While the organization is similar to that of ``PyTorch Lightning``, the goal of ``stable-SSL`` is to significantly reduce codebase complexity without sacrificing performance. Think of ``PyTorch Lightning`` as industry-driven (abstracting everything away), whereas ``stable-SSL`` is academia-driven (providing users with complete visibility into every aspect).


How to launch runs
~~~~~~~~~~~~~~~~~~

.. _launch:

When using ``stable-SSL``, we recommend relying on configuration files to specify the parameters, typically using ``Hydra`` (see `Hydra documentation <https://hydra.cc/>`_).

The parameters are organized into the following groups (more details in the `User Guide <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html>`_):

* **data**: Defines the dataset, loading, and augmentation pipelines. Only the dataset called ``train`` is used for training. If there is no dataset named ``train``, the model runs in evaluation mode. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#data>`_.
* **module**: Specifies the neural network modules, with a required ``backbone`` as the model's core. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#module>`_.
* **loss**: Defines the model's loss function. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#loss>`_.
* **optim**: Contains optimization parameters, including ``epochs``, ``max_steps`` (per epoch), and ``optimizer`` / ``scheduler`` settings. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#optim>`_.
* **hardware**: Specifies the hardware used, including the number of GPUs, CPUs, etc. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#hardware>`_.
* **logger**: Configures model performance monitoring. APIs like `WandB <https://wandb.ai/home>`_ are supported. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#logger>`_.


Then, create a Python script that will load the configuration and launch the run. Here is an example with Hydra:

.. code-block:: python
   :name: run.py

   import hydra
   from omegaconf import OmegaConf

   OmegaConf.register_new_resolver("eval", eval) # to evaluate expressions in the config file

   @hydra.main(version_base="1.2")
   def main(cfg):
       """Load the configuration and launch the run."""
       trainer = hydra.utils.instantiate(
           cfg.trainer, _convert_="object", _recursive_=False
       )
       trainer.setup()
       trainer.launch()


    if __name__ == "__main__":
       main()

In this example, to launch the run using the configuration file ``default_config.yaml`` located in the ``./configs/`` folder, use the following command, where ``run.py`` is the above script: 

.. code-block:: bash

   python run.py -m --config-name default_config --config-path configs/


Examples of Methods
~~~~~~~~~~~~~~~~~~~

+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| **Method**                                       | **Trainer**                                | **Loss**                                 |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| Barlow Twins                                     | `JointEmbeddingTrainer <jointembed_>`_     | `BarlowTwinsLoss <barlow_>`_             |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| BYOL                                             | `SelfDistillationTrainer <selfdistill_>`_  | `NegativeCosineSimilarity <negcosine_>`_ |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| MoCo                                             | `SelfDistillationTrainer <selfdistill_>`_  | `NTXEntLoss <ntxent_>`_                  |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| SimCLR (`config example <exsimclr_>`_)           | `JointEmbeddingTrainer <jointembed_>`_     | `NTXEntLoss <ntxent_>`_                  |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| SimSiam                                          | `SimSiamTrainer <simsiam_>`_               | `NegativeCosineSimilarity <negcosine_>`_ |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+
| VICReg                                           | `JointEmbeddingTrainer <jointembed_>`_     | `VICRegLoss <vicreg_>`_                  |
+--------------------------------------------------+-------------+------------------------------+------------------------------------------+


.. _exsimclr: _github_url/blob/main/examples/simclr_cifar10_full.yaml

.. _ntxent: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.NTXEntLoss.html#stable_ssl.losses.NTXEntLoss
.. _barlow: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.BarlowTwinsLoss.html#stable_ssl.losses.BarlowTwinsLoss
.. _negcosine: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.NegativeCosineSimilarity.html
.. _vicreg: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.losses.VICRegLoss.html

.. _jointembed: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.JointEmbeddingTrainer.html
.. _selfdistill: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.SelfDistillationTrainer.html#stable_ssl.trainers.SelfDistillationTrainer
.. _simsiam: https://rbalestr-lab.github.io/stable-SSL.github.io/dev/gen_modules/stable_ssl.trainers.SimSiamTrainer.html#stable_ssl.trainers.SimSiamTrainer


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