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

At its core, `stable-ssl` provides a [`BaseTrainer`](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer) class that manages job submission, data loading, training, evaluation, logging, monitoring, checkpointing, and requeuing, all customizable via a configuration file. This class is intended to be subclassed for specific training needs (see these [`trainers`](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html) as examples).


## Build a Configuration File

`stable-ssl` uses [`Hydra`](https://hydra.cc/) to manage input parameters through configuration files, enabling efficient hyperparameter tuning with ``multirun`` and seamless integration with job launchers like ``submitit`` for Slurm.

The first step is to specify a **trainer** class which is a subclass of [`BaseTrainer`](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer).
Optionally, the trainer may require a **loss** function which is then used in the `compute_loss` method of the trainer.

The trainer parameters are then structured according to the following categories:

| **Category**     | **Description**                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **data**         | Defines the dataset, loading, and augmentation pipelines. The `train` dataset is used for training, and if absent, the model runs in evaluation mode.  |
| **module**       | Specifies the neural network modules and their architecture.                                                                                           |
| **optim**        | Defines the optimization components, including the optimizer, scheduler, and the number of epochs. See defaults parameters in the [`OptimConfig`].       |
| **hardware**     | Specifies the hardware configuration, including the number of GPUs, CPUs, and precision settings. See defaults parameters in the [`HardwareConfig`].     |
| **logger**       | Configures model performance monitoring. APIs like [`WandB`](https://wandb.ai/home) are supported. See defaults parameters in the [`LoggerConfig`].        |

[`OptimConfig`]: https://rbalestr-lab.github.io/stable-ssl.github.io/dev/api/gen_modules/stable_ssl.config.OptimConfig.html#stable_ssl.config.OptimConfig
[`HardwareConfig`]: https://rbalestr-lab.github.io/stable-ssl.github.io/dev/api/gen_modules/stable_ssl.config.HardwareConfig.html#stable_ssl.config.HardwareConfig
[`LoggerConfig`]: https://rbalestr-lab.github.io/stable-ssl.github.io/dev/api/gen_modules/stable_ssl.config.LoggerConfig.html#stable_ssl.config.LoggerConfig


<details>
  <summary>Config Example : SimCLR CIFAR10</summary>

```yaml
trainer:
  # ===== Base Trainer =====
  _target_: stable_ssl.JointEmbeddingTrainer

  # ===== loss Parameters =====
  loss:
    _target_: stable_ssl.NTXEntLoss
    temperature: 0.5

  # ===== Module Parameters =====
  module:
    backbone:
      _target_: stable_ssl.modules.load_backbone
      name: resnet50
      low_resolution: True
      num_classes: null
    projector:
      _target_: stable_ssl.modules.MLP
      sizes: [2048, 2048, 128]
    projector_classifier:
      _target_: torch.nn.Linear
      in_features: 128
      out_features: ${trainer.data._num_classes}
    backbone_classifier:
      _target_: torch.nn.Linear
      in_features: 2048
      out_features: ${trainer.data._num_classes}

  # ===== Optim Parameters =====
  optim:
    epochs: 1000
    optimizer:
      _target_: stable_ssl.optimizers.LARS
      _partial_: True
      lr: 5
      weight_decay: 1e-6
    scheduler:
      _target_: stable_ssl.schedulers.LinearWarmupCosineAnnealing
      _partial_: True
      total_steps: ${eval:'${trainer.optim.epochs} * ${trainer.data._num_train_samples} // ${trainer.data.train.batch_size}'}

  # ===== Data Parameters =====
  data:
    _num_classes: 10
    _num_train_samples: 50000
    train: # training dataset as indicated by name 'train'
      _target_: torch.utils.data.DataLoader
      batch_size: 256
      drop_last: True
      shuffle: True
      num_workers: 6
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ~/data
        train: True
        download: True
        transform:
          _target_: stable_ssl.data.MultiViewSampler
          transforms:
            # === First View ===
            - _target_: torchvision.transforms.v2.Compose
              transforms:
                - _target_: torchvision.transforms.v2.RandomResizedCrop
                  size: 32
                  scale:
                    - 0.2
                    - 1.0
                - _target_: torchvision.transforms.v2.RandomHorizontalFlip
                  p: 0.5
                - _target_: torchvision.transforms.v2.RandomApply
                  p: 0.8
                  transforms:
                    - {
                        _target_: torchvision.transforms.v2.ColorJitter,
                        brightness: 0.4,
                        contrast: 0.4,
                        saturation: 0.2,
                        hue: 0.1,
                      }
                - _target_: torchvision.transforms.v2.RandomGrayscale
                  p: 0.2
                - _target_: torchvision.transforms.v2.ToImage
                - _target_: torchvision.transforms.v2.ToDtype
                  dtype:
                    _target_: stable_ssl.utils.str_to_dtype
                    _args_: [float32]
                  scale: True
            # === Second View ===
            - _target_: torchvision.transforms.v2.Compose
              transforms:
                - _target_: torchvision.transforms.v2.RandomResizedCrop
                  size: 32
                  scale:
                    - 0.2
                    - 1.0
                - _target_: torchvision.transforms.v2.RandomHorizontalFlip
                  p: 0.5
                - _target_: torchvision.transforms.v2.RandomApply
                  p: 0.8
                  transforms:
                    - {
                        _target_: torchvision.transforms.v2.ColorJitter,
                        brightness: 0.4,
                        contrast: 0.4,
                        saturation: 0.2,
                        hue: 0.1,
                      }
                - _target_: torchvision.transforms.v2.RandomGrayscale
                  p: 0.2
                - _target_: torchvision.transforms.v2.RandomSolarize
                  threshold: 128
                  p: 0.2
                - _target_: torchvision.transforms.v2.ToImage
                - _target_: torchvision.transforms.v2.ToDtype
                  dtype:
                    _target_: stable_ssl.utils.str_to_dtype
                    _args_: [float32]
                  scale: True
    test: # can be any name
      _target_: torch.utils.data.DataLoader
      batch_size: 256
      num_workers: ${trainer.data.train.num_workers}
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

  # ===== Logger Parameters =====
  logger:
    eval_every_epoch: 10
    log_every_step: 100
    wandb: True
    metric:
      test:
        acc1:
          _target_: torchmetrics.classification.MulticlassAccuracy
          num_classes: ${trainer.data._num_classes}
          top_k: 1
        acc5:
          _target_: torchmetrics.classification.MulticlassAccuracy
          num_classes: ${trainer.data._num_classes}
          top_k: 5

  # ===== Hardware Parameters =====
  hardware:
    seed: 0
    float16: true
    device: "cuda:0"
    world_size: 1

```
</details>


## Launch a Run

To launch a run using a configuration file located in a specified folder, simply use the following command:

```bash
stable-ssl --config-path <config_path> --config-name <config_name>
```

Replace `<config_path>` with the path to your configuration folder and `<config_name>` with the name of your configuration file.

Useful options include:

<details>
  <summary>Launching in multirun</summary>

```bash
stable-ssl --multirun --config-path <config_path> --config-name <config_name> ++trainer.data.train.batch_size=128,256,512
```
</details>

<details>
  <summary>Launching on slurm</summary>

```bash
stable-ssl --multirun --config-path <config_path> --config-name <config_name> hydra/launcher=submitit_slurm
```
</details>

> **Note:**
> One must include the `--multirun` flag when using a launcher like `submitit_slurm`.


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```bash
pip install -e .
```

Or you can also run:

```bash
pip install -U git+https://github.com/rbalestr-lab/stable-ssl
```

## Ways You Can Contribute:

- If you'd like to contribute new features, bug fixes, or improvements to the documentation, please refer to our [contributing guide](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/contributing.html) for detailed instructions on how to get started.

- You can also contribute by adding new methods, datasets, or configurations that improve the current performance of a method in the [benchmark section](https://github.com/rbalestr-lab/stable-ssl/tree/main/benchmarks).
