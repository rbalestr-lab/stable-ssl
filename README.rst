.. image:: https://github.com/rbalestr-lab/stable-SSL/raw/main/docs/source/figures/logo.jpg
   :alt: ssl logo
   :width: 200px
   :align: right

|Documentation| |Benchmark| |CircleCI| |Pytorch| |Black| |License| |WandB|


⚠️ This library is currently in a phase of active development. All features are subject to change without prior notice.


The Self-Supervised Learning Library by Researchers for Researchers
===================================================================

*Have a research idea? With stable-SSL, you can go from concept to execution in under 10 minutes. Start from scratch and quickly set up your pipeline, all while being able to generate high-quality figures and tables from your results. That's the goal of stable-SSL.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: PytorchLightning, VISSL, WandB, Hydra, Submitit.

``stable-SSL`` implements all the basic boilerplate code, including data loading, logging, checkpointing and optimization. It offers users full flexibility to customize each part of the pipeline through a configuration file, enabling easy selection of network architectures, loss functions, evaluation metrics, data augmentations and more.
These components can be sourced from stable-SSL itself, popular libraries like PyTorch, or custom modules created by the user.


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
At its core, ``stable-SSL`` provides a ``BaseModel`` class that sequentially calls the following methods:

.. code-block:: text

   - self.before_fit (nothing by default)
   - self.fit (executes all the training/intermitent evaluation by default)
      - for `self.optim["epochs"]` epochs:
         - self.fit_epoch (one training epoch by default)
            - self.before_fit_epoch (setup in train mode)
            - loop over mini-batches
               - self.before_fit_step (moves data to device)
               - self.fit_step (computes loss and performs optimization step)
               - self.after_fit_step (nothing by default)
            - self.after_fit_epoch (nothing by default)
         - self.evaluate (if asked by user config, looping over all non train datasets)
            - self.before_eval (setup in eval mode)
            - loop over mini-batches
               - self.before_eval_step (moves data to device)
               - self.eval_step (computes eval metrics)
               - self.after_eval_step (nothing by default)
            - self.after_eval (nothing by default)
         - save intermitent checkpoint if asked by user config
      - save final checkpoint if asked by user config
   - self.after_fit (evaluates by default)

While the organization is similar to that of ``PyTorch Lightning``, the goal of ``stable-SSL`` is to significantly reduce codebase complexity without sacrificing performance. Think of ``PyTorch Lightning`` as industry-driven (abstracting everything away), whereas ``stable-SSL`` is academia-driven (providing users with complete visibility into every aspect).


How to launch runs
~~~~~~~~~~~~~~~~~~

.. _launch:

When using ``stable-SSL``, we recommend relying on configuration files to specify the parameters, typically using ``Hydra`` (see `Hydra documentation <https://hydra.cc/>`_).

The parameters are organized into the following groups (more details in the `User Guide <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html>`_):

* ``data``: Defines the dataset, loading, and augmentation pipelines. Only the dataset called ``train`` is used for training. If there is no dataset named ``train``, the model runs in evaluation mode. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#data>`_.
* ``modules``: Specifies the neural network modules, with a required ``backbone`` as the model's core. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#module>`_.
* ``objective``: Defines the model's loss function. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#objective>`_.
* ``optim``: Contains optimization parameters, including ``epochs``, ``max_steps`` (per epoch), and ``optimizer`` / ``scheduler`` settings. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#optim>`_.
* ``hardware``: Specifies the hardware used, including the number of GPUs, CPUs, etc. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#hardware>`_.
* ``logger``: Configures model performance monitoring. APIs like `WandB <https://wandb.ai/home>`_ are supported. `Example <https://rbalestr-lab.github.io/stable-SSL.github.io/dev/user_guide.html#logger>`_.


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

   python run.py --config-name default_config --config-path configs/


Examples of Methods
~~~~~~~~~~~~~~~~~~

+---------------+-----------+-------------------+--------------------------+
| Methods       | Predictor | Self-distillation | Loss                     |
+---------------+-----------+-------------------+--------------------------+
| Barlow Twins  | ❌        | ❌                | BarlowTwinsLoss          |
+---------------+-----------+-------------------+--------------------------+
| BYOL          | ✅        | ✅                | NegativeCosineSimilarity |
+---------------+-----------+-------------------+--------------------------+
| MoCo          | ❌        | ✅                | NTXEntLoss               |
+---------------+-----------+-------------------+--------------------------+
| SimCLR        | ❌        | ❌                | NTXEntLoss               |
+---------------+-----------+-------------------+--------------------------+
| SimSiam       | ✅        | ❌                | NegativeCosineSimilarity |
+---------------+-----------+-------------------+--------------------------+
| VICReg        | ✅        | ✅                | VICRegLoss               |
+---------------+-----------+-------------------+--------------------------+




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