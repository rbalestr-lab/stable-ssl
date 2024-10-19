Stable-SSL: the Self-Supervised Learning Library by Researchers for Researchers
===============================================================================

*You got a research idea? It shouldn't take you more than 10 minutes to start from scratch and get it running with the ability to produce high quality figures/tables from the results: that's the goal of `stable-SSL`.*

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: PytorchLightning, VISSL, Wandb, Hydra, Submitit.

.. image:: ./assets/logo.jpg
   :alt: ssl logo
   :width: 200px
   :align: right

.. contents:: Table of Contents
   :depth: 2


Why stable-SSL?
---------------

.. _why:

A quick search of `AI libraries` or `Self Supervised Learning libraries` will return hundreds of results. 99% will be independent project-centric libraries that can't be reused for general purpose AI research. The other 1% includes:

- Framework libraries such as PytorchLightning that focus on production needs
- SSL libraries such as VISSL, FFCV-SSL, LightlySSL that are too rigid, often discontinued or not maintained, or commercial
- Standalone libraries such as Wandb, submitit, Hydra that do not offer enough boilerplate for AI research

Hence our goal is to fill that void.

How stable-SSL helps you
------------------------

.. _how:

`stable-SSL` implements all the basic boilerplate code, including data loader, logging, checkpointing, optimization, etc. You only need to implement 3 methods to get started: your loss, your model, and your prediction (see `example <#own_trainer>`_ below). But if you want to customize more things, simply inherit the base `Trainer` and override any method! This could include different metrics, different data samples, different training loops, etc.

Installation
------------

.. _installation:

The library is not yet available on PyPI. You can install it from the source code, as follows.

.. code-block:: bash

   pip install -e .

Or you can also run:

.. code-block:: bash

   pip install git+https://github.com/rbalestr-lab/stable-SSL

Minimal Documentation
---------------------

.. _minimal:

Implement your own `Trainer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _own_trainer:

At the very least, you need to implement three methods:

- `initialize_modules`: this method initializes whatever model and parameters to use for training/inference
- `forward`: that method that will be doing the prediction, e.g., for classification it will be p(y|x)
- `compute_loss`: that method should return a scalar value used for backpropagation/training.

Pass user arguments
~~~~~~~~~~~~~~~~~~~

.. _arguments:

To pass a user argument e.g., `my_arg` that is not already supported in our configs (i.e., different than `optim.lr` etc.), there are two options:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **With Hydra**
     - **Without Hydra**
   * - Pass your argument when calling the Python script as `++my_arg=2`

       .. code-block:: python

          @hydra.main(version_base=None)
          def main(cfg: DictConfig):
              args = ssl.get_args(cfg)
              args.my_arg  # your arg!
              trainer = MyTrainer(args)
              trainer.config.my_arg  # your arg!

     - Pass your argument to your `Trainer`

       .. code-block:: python

          @hydra.main(version_base=None)
          def main(cfg: DictConfig):
              args = ssl.get_args(cfg)
              trainer = MyTrainer(args, my_arg=2)
              trainer.config.my_arg  # your arg!

Your argument can be retrieved anywhere inside your `Trainer` instance through `self.config.my_arg` with either of the two above options.

Write and Read your logs (Wandb or JSON)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _logs:

- **Loggers**: We support the `*Weights and Biases* <https://wandb.ai/site>`_ and `*jsonlines* <https://jsonlines.readthedocs.io/en/latest/>`_ for logging. For the Wandb, you will need to use the following tags: `log.entity` (optional), `log.project` (optional), `log.run` (optional). They are all optional since Wandb handles its own exceptions if those are not passed by users. For jsonlines, the `log.folder` / `log.name` is where the logs will be dumped. Both are also optional. `log.folder` will be set to `./logs` and `log.name` will be set to `%Y%m%d_%H%M%S.%f` of the call. References: `stable_ssl.configs.LogConfig`, `stable_ssl.configs.WandbConfig`.

- **Logging values**: we have a unified logging framework regardless of the logger you employ. You can directly use `self.log({"loss": 0.001, "lr": 1})` which will add an entry or row in Wandb or the text file. If you want to log many different things at once, it can be easier to "pack" your log commits, as in:

  .. code-block:: python

     self.log({"loss": 0.001}, commit=False)
     ...
     self.log({"lr": 1})

  `stable-SSL` will automatically pack those two and commit the logs.

- **Reading logs (Wandb):**

  .. code-block:: python

     from stable_ssl import reader

     # single run
     config, df = reader.wandb_run(
         ENTITY_NAME, PROJECT_NAME, RUN_NAME
     )

     # single project (multiple runs)
     configs, dfs = reader.wandb_project(ENTITY_NAME, PROJECT_NAME)

- **Reading logs (jsonl):**

  .. code-block:: python

     from stable_ssl import reader

     # single run
     config, df = reader.jsonl_run(
         FOLDER_NAME, RUN_NAME
     )
     # single project (multiple runs)
     configs, dfs = reader.jsonl_project(FOLDER_NAME)

- **Reading logs (json+CLI):**

  .. code-block:: bash

     python cli/plot_metric.py --path PATH --metric eval/epoch/acc1 --savefig ./test.png --hparams model.name,optim.lr

Multi-run
~~~~~~~~~

.. _multirun:

To launch multiple runs, add `-m` and specify the multiple values to try as `++group.variable=value1,value2,value3`. For instance:

.. code-block:: bash

   python3 main.py --config-name=simclr_cifar10_sgd -m ++optim.lr=2,5,10

Slurm
~~~~~

.. _slurm:

To launch on slurm simply add `hydra/launcher=submitit_slurm` in the command line, for instance:

.. code-block:: bash

   python3 main.py hydra/launcher=submitit_slurm hydra.launcher.timeout_min=3

**Remark**: All the parameters of the slurm `hydra.launcher` are given `here <https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py>`_ (similar to submitit).

Or to specify the slurm launcher you can add in the config file:

.. code-block:: yaml

   defaults:
     - override hydra/launcher: submitit_slurm

Library Design
~~~~~~~~~~~~~~

.. _design:

Stable-SSL provides all the boilerplate to quickly get started doing AI research, with a focus on Self Supervised Learning (SSL) albeit other applications can certainly build upon Stable-SSL. In short, we provide a `BaseModel` class that calls the following methods (in order):

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

The file `main.py` to launch experiments is located in the `runs/` folder.

The default parameters are given in the `sable_ssl/config.py` file.
The parameters are structured in the following groups: data, model, hardware, log, optim.

Using default config files
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use default config files that are located in `runs/configs`. To do so, simply specify the config file with the `--config-name` command as follows:

.. code-block:: bash

   python3 train.py --config-name=simclr_cifar10_sgd --config-path configs/

Classification case
~~~~~~~~~~~~~~~~~~~

- **How is the accuracy calculated?** The predictions are assumed to be the output of the forward method, then this is fed into a few metrics along with `self.data[1]` which is assumed to encode the labels.

Setting params in command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can modify/add parameters of the config file by adding `++group.variable=value` as follows:

.. code-block:: bash

   python3 main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
   # same but with SLURM
   python3 main.py --config-name=simclr_cifar10_sgd ++optim.epochs=4 ++optim.lr=1 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=1800 hydra.launcher.cpus_per_task=4 hydra.launcher.gpus_per_task=1 hydra.launcher.partition=gpu-he

**Remark**: If `group.variable` is already in the config file you can use `group.variable=value` and if it is not you can use `+group.variable=value`. The `++` command handles both cases; that's why I would recommend using it.
