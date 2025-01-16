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

At its core, `stable-ssl` provides a [BaseTrainer](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/gen_modules/stable_ssl.BaseTrainer.html#stable_ssl.BaseTrainer) class that manages job submission, data loading, training, evaluation, logging, monitoring, checkpointing, and requeuing, all customizable via a configuration file. This class is intended to be subclassed for specific training needs (see these [trainers](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/trainers.html) as examples).

`stable-ssl` leverages [`Hydra`](https://hydra.cc/) to manage input parameters through configuration files, offering benefits like efficient hyperparameter experimentation with `multirun` and smooth integration with job launchers such as `submitit` for Slurm.


## Build a Configuration File

In `stable-ssl`, the configuration file is structured according to the following categories:

### trainer

Specifies the trainer class. It is a subclass of `BaseTrainer`.

<details>
  <summary>Example</summary>

```yaml
trainer:
  _target_: stable_ssl.JointEmbeddingTrainer
```
</details>

### data

Defines the dataset, loading, and augmentation pipelines. Only the dataset called `train` is used for training. If there is no dataset named `train`, the model runs in evaluation mode.

<details>
  <summary>Example</summary>

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


### module

Specifies the neural network modules and their architecture.

<details>
  <summary>Example</summary>

```yaml
module:
   backbone:
      _target_: stable_ssl.modules.load_backbone
      name: resnet18
      low_resolution: True
      num_classes: null
   projector:
      _target_: torch.nn.Sequential
      _args_:
         - _target_: torch.nn.Linear
            in_features: 512
            out_features: 2048
            bias: False
         - _target_: torch.nn.BatchNorm1d
            num_features: ${trainer.module.projector._args_.0.out_features}
         - _target_: torch.nn.ReLU
         - _target_: torch.nn.Linear
            in_features: ${trainer.module.projector._args_.0.out_features}
            out_features: 128
            bias: False
         - _target_: torch.nn.BatchNorm1d
            num_features: ${trainer.module.projector._args_.3.out_features}
   projector_classifier:
      _target_: torch.nn.Linear
      in_features: 128
      out_features: ${trainer.data._num_classes}
   backbone_classifier:
      _target_: torch.nn.Linear
      in_features: 512
      out_features: ${trainer.data._num_classes}
```
</details>

### optim

Defines all the components needed for optimization, including the optimizer, scheduler, and the number of epochs.

<details>
  <summary>Example</summary>

```yaml
optim:
 epochs: 1000
 optimizer:
   _target_: stable_ssl.optimizers.LARS
   _partial_: True
   lr: 5
   weight_decay: 1e-6
 scheduler:
   _target_: stable_ssl.scheduler.LinearWarmupCosineAnnealing
   _partial_: True
   total_steps: ${eval:'${trainer.optim.epochs} * ${trainer.data._num_samples} // ${trainer.data.train.batch_size}'}
```
</details>


### hardware

Specifies the hardware used, including the number of GPUs, CPUs, etc.

<details>
  <summary>Example</summary>

```yaml
hardware:
   seed: 0
   float16: true
   device: "cuda:0"
   world_size: 1
```
</details>

### Logger

Configures model performance monitoring. APIs like [WandB](https://wandb.ai/home) are supported.

<details>
  <summary>Example</summary>

```yaml
logger:
   wandb: true
   base_dir: "./"
   level: 20
   checkpoint_frequency: 1
   log_every_step: 1
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
