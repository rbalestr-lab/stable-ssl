# stable-ssl

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/rbalestr-lab/stable-ssl/tree/main/benchmarks)
[![Test Status](https://github.com/rbalestr-lab/stable-ssl/actions/workflows/testing.yml/badge.svg)](https://github.com/rbalestr-lab/stable-ssl/actions/workflows/testing.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-ssl/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-ssl/tree/main)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)

⚠️ **This library is currently in a phase of active development. All features are subject to change without prior notice.**

``stable-ssl`` streamlines training self-supervised learning models by offering all the essential boilerplate code with minimal hardcoded utilities. Its flexible and modular design allows seamless integration of components from external libraries, including architectures, loss functions, evaluation metrics, and augmentations.

At its core, `stable-ssl` provides a [`BaseTrainer`](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer) class that manages job submission, data loading, training, evaluation, logging, monitoring, checkpointing, and requeuing, all customizable via a configuration file. This class is intended to be subclassed for specific training needs (see these [trainers](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html) as examples).


## Build a Configuration File

`stable-ssl` uses [`Hydra`](https://hydra.cc/) to manage input parameters through configuration files, enabling efficient hyperparameter tuning with ``multirun`` and seamless integration with job launchers like ``submitit`` for Slurm.

The first step is to specify a **trainer** class which is a subclass of `BaseTrainer`.
Optionally, the trainer may require a **loss** function which is then used in the `compute_loss` method of the trainer.

The trainer parameters are then structured according to the following categories:

| **Category**     | **Description**                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **data**         | Defines the dataset, loading, and augmentation pipelines. The `train` dataset is used for training, and if absent, the model runs in evaluation mode.  |
| **module**       | Specifies the neural network modules and their architecture.                                                                                           |
| **optim**        | Defines the optimization components, including the optimizer, scheduler, and the number of epochs. See defaults parameters in the [OptimConfig]        |
| **hardware**     | Specifies the hardware configuration, including the number of GPUs, CPUs, and precision settings.                                                      |
| **logger**       | Configures model performance monitoring. APIs like [WandB](https://wandb.ai/home) are supported                                                        |

[OptimConfig]: https://rbalestr-lab.github.io/stable-ssl.github.io/dev/api/gen_modules/stable_ssl.config.OptimConfig.html#stable_ssl.config.OptimConfig


<details>
  <summary>Config Example : SimCLR CIFAR10</summary>

```yaml
data:
  _num_classes: 10
  _num_samples: 50000
  train:
    _target_: torch.utils.data.DataLoader
    batch_size: 256
    drop_last: True
    shuffle: True
    num_workers: ${trainer.hardware.cpus_per_task}
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ~/data
      train: True
      transform:
        _target_: stable_ssl.data.MultiViewSampler
        transforms:
          - _target_: torchvision.transforms.v2.Compose
            transforms:
              - _target_: torchvision.transforms.v2.RandomResizedCrop
                size: 32
                scale:
                  - 0.2
                  - 1.0
              - _target_: torchvision.transforms.v2.RandomHorizontalFlip
                p: 0.5
              - _target_: torchvision.transforms.v2.ToImage
              - _target_: torchvision.transforms.v2.ToDtype
                dtype:
                  _target_: stable_ssl.utils.str_to_dtype
                  _args_: [float32]
                scale: True
          - ${trainer.data.base.dataset.transform.transforms.0}
  test:
    _target_: torch.utils.data.DataLoader
    batch_size: 256
    num_workers: ${trainer.hardware.cpus_per_task}
    dataset:
      _target_: torchvision.datasets.CIFAR10
      train: False
      root: ~/data
      transform:
        _target_: torchvision.transforms.v2.Compose
        transforms:
          - _target_: torchvision.transforms.v2.ToImage
          - _target_: torchvision.transforms.v2.ToDtype
            dtype:
              _target_: stable_ssl.utils.str_to_dtype
              _args_: [float32]
            scale: True
```
</details>


## Launch a run

To launch a run using a configuration file located in a specified folder, simply use the following command:

```bash
stable-ssl --config-path <config_path> --config-name <config_name>
```

Replace `<config_path>` with the path to your configuration folder and `<config_name>` with the name of your configuration file.


This command utilizes [Hydra](https://hydra.cc/), making it compatible with multirun functionality and CLI overrides. It is important to note that the multirun flag (`-m` or `--multirun`) is **mandatory** when using the Slurm launcher.


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```bash
pip install -e .
```

Or you can also run:

```bash
pip install -U git+https://github.com/rbalestr-lab/stable-ssl
```
