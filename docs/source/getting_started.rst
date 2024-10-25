.. _getting_started:

.. currentmodule:: stable_ssl

.. automodule:: stable_ssl
   :no-members:
   :no-inherited-members:


Getting Started
================

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


How to launch runs
------------------

.. _launch:

First build a confif file with the parameters you want to use. The parameters should be structured in the following groups: data, model, hardware, log, optim.
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


.. Classification case
.. ~~~~~~~~~~~~~~~~~~~

.. - **How is the accuracy calculated?** The predictions are assumed to be the output of the forward method, then this is fed into a few metrics along with ``self.data[1]`` which is assumed to encode the labels.


Advanced Usage
--------------

Setting params in command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can modify/add parameters of the config file by adding ``++group.variable=value`` as follows:

.. code-block:: bash

   python3 main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
   
.. # same but with SLURM
.. python3 main.py --config-name=simclr_cifar10_sgd ++optim.epochs=4 ++optim.lr=1 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=1800 hydra.launcher.cpus_per_task=4 hydra.launcher.gpus_per_task=1 hydra.launcher.partition=gpu-he

**Remark**: If ``group.variable`` is already in the config file you can use ``group.variable=value`` and if it is not you can use ``+group.variable=value``. The ``++`` command handles both cases; thus we recommend using it.


Pass user arguments
~~~~~~~~~~~~~~~~~~~

.. _arguments:

To pass a user argument e.g., ``my_arg`` that is not already supported in our configs (i.e., different than ``optim.lr`` etc.), there are two options:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **With Hydra**
     - **Without Hydra**
   * - Pass your argument when calling the Python script as ``++my_arg=2``

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

Your argument can be retrieved anywhere inside your ``Trainer`` instance through ``self.config.my_arg`` with either of the two above options.


Write and Read your logs (Wandb or JSON)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _logs:

- **Loggers**: We support the `Weights and Biases <https://wandb.ai/site>`_ and `jsonlines <https://jsonlines.readthedocs.io/en/latest/>`_ for logging. For the Wandb, you will need to use the following tags: ``log.entity`` (optional), ``log.project`` (optional), ``log.run`` (optional). They are all optional since Wandb handles its own exceptions if those are not passed by users. For jsonlines, the ``log.folder`` / ``log.name`` is where the logs will be dumped. Both are also optional. ``log.folder`` will be set to ``./logs`` and ``log.name`` will be set to ``%Y%m%d_%H%M%S.%f`` of the call. References: ``stable_ssl.configs.LogConfig``, ``stable_ssl.configs.WandbConfig``.

- **Logging values**: we have a unified logging framework regardless of the logger you employ. You can directly use ``self.log({"loss": 0.001, "lr": 1})`` which will add an entry or row in Wandb or the text file. If you want to log many different things at once, it can be easier to "pack" your log commits, as in:

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

To launch multiple runs, add `-m` and specify the multiple values to try as ``++group.variable=value1,value2,value3``. For instance:

.. code-block:: bash

   python3 main.py --config-name=simclr_cifar10_sgd -m ++optim.lr=2,5,10

Slurm
~~~~~

.. _slurm:

To launch on slurm simply add ``hydra/launcher=submitit_slurm`` in the command line, for instance: 

.. code-block:: bash

   python3 main.py hydra/launcher=submitit_slurm hydra.launcher.timeout_min=3

**Remark**: All the parameters of the slurm ``hydra.launcher`` are given `here <https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py>`_ (similar to submitit).

Or to specify the slurm launcher you can add in the config file:

.. code-block:: yaml

   defaults:
     - override hydra/launcher: submitit_slurm

