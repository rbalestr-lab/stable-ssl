# Config-Based Training with stable-pretraining

This guide explains how to use Hydra configs to launch training runs with stable-pretraining.

## Quick Start

### Running the Example Config

```bash
# Run SimCLR training on CIFAR-10
python -m stable_pretraining.train --config-path ../examples --config-name simclr_cifar10_config
```

### Overriding Parameters

You can override any parameter from the command line:

```bash
# Change learning rate and epochs
python -m stable_pretraining.train \
    --config-path ../examples \
    --config-name simclr_cifar10_config \
    module.optim.optimizer.lr=0.01 \
    trainer.max_epochs=200

# Use different backbone
python -m stable_pretraining.train \
    --config-path ../examples \
    --config-name simclr_cifar10_config \
    module.backbone.name=resnet50

# Change batch size
python -m stable_pretraining.train \
    --config-path ../examples \
    --config-name simclr_cifar10_config \
    data.train.batch_size=512
```

## Config Structure

The config file (`simclr_cifar10_config.yaml`) contains all components needed for training:

### 1. **Trainer Configuration**
```yaml
trainer:
  _target_: lightning.Trainer
  max_epochs: 1000
  accelerator: gpu
  devices: 1
```

### 2. **Module Configuration**
```yaml
module:
  _target_: stable_pretraining.Module
  forward:
    _target_: stable_pretraining.forward_functions.simclr_forward
  backbone:
    _target_: stable_pretraining.backbone.from_torchvision
    name: resnet18
```

### 3. **Data Configuration**
```yaml
data:
  _target_: stable_pretraining.data.DataModule
  train:
    _target_: torch.utils.data.DataLoader
    batch_size: 256
```

### 4. **Callbacks Configuration**

Callbacks are configured as a list under `trainer.callbacks`:

```yaml
trainer:
  callbacks:
    # Linear probe for online evaluation
    - _target_: stable_pretraining.callbacks.OnlineProbe
      name: linear_probe
      input: embedding
      target: label
      probe:
        _target_: torch.nn.Linear
        in_features: 512
        out_features: 10

    # KNN evaluation
    - _target_: stable_pretraining.callbacks.OnlineKNN
      name: knn_probe
      input: embedding
      target: label
      k: 10

    # LiDAR monitoring
    - _target_: stable_pretraining.callbacks.LiDAR
      name: lidar
      input: embedding
      n: 128
```

## Available Callbacks

stable-pretraining provides many callbacks that can be configured via YAML:

- **OnlineProbe**: Linear evaluation during training
- **OnlineKNN**: K-nearest neighbors evaluation
- **LiDAR**: Representation quality monitoring
- **RankMe**: Feature diversity measurement
- **EarlyStopping**: Stop training based on metrics
- **ModuleSummary**: Model architecture summary
- **ImageRetrieval**: Image retrieval evaluation
- **TeacherStudentCallback**: For distillation setups

## Creating Your Own Config

1. **Copy the example config**:
```bash
cp examples/simclr_cifar10_config.yaml examples/my_experiment.yaml
```

2. **Modify components as needed**:
- Change the model architecture
- Use different loss functions
- Add or remove callbacks
- Adjust hyperparameters

3. **Run your experiment**:
```bash
python -m stable_pretraining.train \
    --config-path ../examples \
    --config-name my_experiment
```

## Advanced Features

### Hyperparameter Sweeps

Use Hydra's multirun feature:

```bash
python -m stable_pretraining.train --multirun \
    --config-path ../examples \
    --config-name simclr_cifar10_config \
    module.optim.optimizer.lr=0.001,0.01,0.1 \
    module.simclr_loss.temperature=0.1,0.5,1.0
```

### Using Different Forward Functions

The example includes several pre-defined forward functions:

- `simclr_forward`: For SimCLR training
- `byol_forward`: For BYOL training
- `vicreg_forward`: For VICReg training
- `barlow_twins_forward`: For Barlow Twins
- `supervised_forward`: For supervised training

Change the forward function in your config:

```yaml
module:
  forward:
    _target_: stable_pretraining.forward_functions.byol_forward
```

### Custom Forward Functions

You can also define custom forward functions:

```python
# my_forward.py
def custom_forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    # Your custom logic here
    return out
```

Then reference it in config:

```yaml
module:
  forward:
    _target_: my_forward.custom_forward
```

## Tips

1. **Everything in One File**: The example shows a complete config in one file for simplicity. You can modularize if needed using Hydra's defaults system.

2. **Instantiation**: The `_target_` field tells Hydra which class/function to instantiate. All nested configs are recursively instantiated.

3. **References**: Use `${...}` syntax to reference other parts of the config and avoid duplication.

4. **Debugging**: The training script prints the full resolved config before training starts.

## Comparison with Python Scripts

### Before (Python Script - 150+ lines):
```python
import torch
import lightning as pl
import stable_pretraining as spt

# Define transforms (20+ lines)
transform = ...

# Create datasets (10+ lines)
dataset = ...

# Create dataloaders (10+ lines)
dataloader = ...

# Define model components (30+ lines)
backbone = ...
projector = ...

# Define forward function (20+ lines)
def forward(self, batch, stage):
    ...

# Create module (10+ lines)
module = spt.Module(...)

# Setup callbacks (20+ lines)
callbacks = [...]

# Create trainer (10+ lines)
trainer = pl.Trainer(...)

# Run training
manager = spt.Manager(...)
manager()
```

### After (YAML Config):
```yaml
# Complete config in one file
# Just run: python -m stable_pretraining.train --config-name my_config
```

The config approach provides:
- ✅ Better reproducibility (configs saved with runs)
- ✅ Easier experimentation (CLI overrides)
- ✅ Cleaner organization (declarative vs imperative)
- ✅ Built-in sweep support
- ✅ No code duplication
