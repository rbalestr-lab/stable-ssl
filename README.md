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

``stable-ssl`` streamlines training self-supervised learning models by offering all the essential boilerplate code with minimal hardcoded utilities. Its modular and flexible design supports seamless integration of architectures, loss functions, evaluation metrics, augmentations, and more from any source.

At its core, `stable-ssl` provides a [BaseTrainer](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer) class that handles job submission, data loading, model training, evaluation, logging, monitoring, checkpointing, and requeuing. Every component is fully customizable through a configuration file. This class is intended to be subclassed for specific training needs (see these [trainers](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html) as examples).


## Launch a run

`stable-ssl` uses `Hydra` (see the [Hydra documentation](https://hydra.cc/)) to manage input parameters via configuration files. These parameters are grouped into the following categories (detailed in the [User Guide](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html)):

To start a run using the `default_config.yaml` configuration file located in the `./configs/` folder, use the following command:

```bash
stable-ssl --config-path configs/ --config-name default_config
```

This command utilizes [Hydra](https://hydra.cc/), making it compatible with multirun functionality and CLI overrides. It is important to note that the multirun flag (`-m` or `--multirun`) is **mandatory** when using the Slurm launcher.



## How to Build a Configuration File


### Data
Defines the dataset, loading, and augmentation pipelines. Only the dataset called `train` is used for training. If there is no dataset named `train`, the model runs in evaluation mode. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#data).

<details>
  <summary>Example data YAML (click to reveal)</summary>

```yaml
trainer:
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


### Module
Specifies the neural network modules. For instance: `backbone`, `projector`, etc. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#module).

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

### Optim
Contains optimization parameters, including `epochs`, `max_steps` (per epoch), and `optimizer` / `scheduler` settings. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#optim).

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

### Hardware
Specifies the hardware used, including the number of GPUs, CPUs, etc. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#hardware).

<details>
  <summary>Example hardware YAML (click to reveal)</summary>

```yaml
hardware:
  gpus: 1
  cpus: 8
  precision: 16
```
</details>

### Logger
Configures model performance monitoring. APIs like [WandB](https://wandb.ai/home) are supported. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#logger).

<details>
  <summary>Example logger YAML (click to reveal)</summary>

```yaml
logger:
  name: "wandb"
  project: "my_ssl_experiment"
  entity: "my_username"
```
</details>

### Loss (optional)
Defines a loss function that can then be used in the `compute_loss` method of the trainer. [Example](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/user_guide.html#loss).

<details>
  <summary>Example loss YAML (click to reveal)</summary>

```yaml
loss:
  name: "NTXEntLoss"
  temperature: 0.5
```
</details>


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```bash
pip install -e .
```

Or you can also run:

```bash
pip install -U git+https://github.com/rbalestr-lab/stable-ssl
```
