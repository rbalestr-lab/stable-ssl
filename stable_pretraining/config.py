"""Configuration classes specifying default parameters for stable-SSL."""

import logging
import lzma
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig


def collapse_nested_dict(
    cfg: Union[dict, object],
    level_separator: str = ".",
    _base_name: str = None,
    _flat_cfg: dict = None,
) -> dict:
    """Parse a Hydra config and make it readable for wandb (flatten).

    Args:
        cfg (Union[dict, object]): The original (Hydra) nested dict.
        level_separator (str, optional): The string to separate level names. Defaults to ".".
        _base_name (str, optional): The parent string, used for recursion only, users should ignore.
            Defaults to None.
        _flat_cfg (dict, optional): The flattened config, used for recursion only, users should ignore.
            Defaults to None.

    Returns:
        dict: Flat config.
    """
    # INIT
    if _flat_cfg is None:
        _flat_cfg = {}
    if _base_name is None:
        _base_name = ""
    if isinstance(cfg, list) or isinstance(cfg, tuple):
        for i in range(len(cfg)):
            collapse_nested_dict(
                cfg[i],
                level_separator=level_separator,
                _base_name=_base_name + f"{level_separator}{i}",
                _flat_cfg=_flat_cfg,
            )
    elif isinstance(cfg, dict) or isinstance(cfg, omegaconf.dictconfig.DictConfig):
        for key in cfg:
            collapse_nested_dict(
                cfg[key],
                level_separator=level_separator,
                _base_name=_base_name + f"{level_separator}{key}",
                _flat_cfg=_flat_cfg,
            )
    else:
        if _base_name.startswith(level_separator):
            _base_name = _base_name[len(level_separator) :]
        _flat_cfg[_base_name] = cfg
    return _flat_cfg


def instanciate_config(cfg=None, debug_hash=None) -> object:
    """Instantiate the config and debug hash."""
    if debug_hash is None:
        assert cfg is not None
        print("Your debugging hash:", lzma.compress(pickle.dumps(cfg)))
    else:
        print("Using debugging hash")
        cfg = pickle.loads(lzma.decompress(debug_hash))
    trainer = hydra.utils.instantiate(
        cfg.trainer, _convert_="object", _recursive_=False
    )
    for key, value in cfg.items():
        if key == "trainer":
            continue
        logging.info(f"\t=> Adding user arg {key} to Trainer")
        if hasattr(trainer, key):
            raise ValueError(f"User arg {key} already exists in the Trainer {trainer}")
        setattr(trainer, key, value)
    return trainer


@dataclass
class HardwareConfig:
    """Configuration for the hardware parameters.

    Args:
        seed (int, optional): Random seed for reproducibility. Default is None.
        float16 (bool, optional): Whether to use mixed precision (float16) for training.
            Default is False.
        world_size (int, optional): Number of processes participating in distributed training.
            Default is 1.
        device (str, optional): The device to use for training. Default is "cuda" if available, else "cpu".
    """

    seed: Optional[int] = None
    float16: bool = False
    world_size: int = 1
    device: str = "cuda"


@dataclass
class LoggerConfig:
    """Configuration for logging and checkpointing during training or evaluation.

    Args:
        level (int, optional): The logging level. Determines the threshold for what gets logged. Default is 20.
        metric (dict, optional): A dictionary to store and log various metrics. Default is an empty dict.
        monitor (dict, optional): A dictionary to store and log various monitoring statistics.
            Default is an empty dict
        save_final_model (str or bool, optional): Specifies whether to save the final trained model.
            If a name is provided, the final model will be saved with that name.
            Default is False.
        eval_every_epoch (int, optional): The frequency (in epochs) at which the model will be evaluated.
            For example, if set to 1, evaluation occurs every epoch. Default is 1.
        log_every_step (int, optional): The frequency (in training steps) at which to log intermediate metrics.
            For example, if set to 1, logs occur every step. Default is 1.
        checkpoint_frequency (int, optional): The frequency (in epochs) at which model checkpoints are saved.
            For example, if set to 10, a checkpoint is saved every 10 epochs.
            Default is None.
        checkpoint_model_only (bool, optional): Whether to save only the model weights (True) or save additional training state
            (False) during checkpointing. Default is True.
        dump_path (pathlib.Path, optional): The path where output is dumped. Defaults to Hydra's runtime output directory.
        wandb (bool or dict or None, optional): Configuration for Weights & Biases logging.
            If `True`, it will be converted to an empty dictionary and default keys will be
            filled in if `rank == 0`. Default is None.
            See :mod:`stable_pretraining.config.WandbConfig`
            for the full list of parameters and their defaults.
    """

    level: int = 20
    metric: dict = field(default_factory=dict)
    monitor: dict = field(default_factory=dict)
    save_final_model: Union[str, bool] = False
    eval_every_epoch: int = 1
    log_every_step: int = 1
    checkpoint_frequency: Optional[int] = None
    checkpoint_model_only: bool = True
    dump_path: Path = field(
        default_factory=lambda: Path(HydraConfig.get().runtime.output_dir)
    )
    wandb: Union[bool, dict, None] = None


@dataclass
class WandbConfig:
    """Configuration for the Weights & Biases logging.

    Args:
        dir (pathlib.Path, optional): The path where output is dumped. Defaults to Hydra's runtime output directory.
        entity (str, optional): Name of the (Weights & Biases) entity. Default is None.
        project (str, optional): Name of the (Weights & Biases) project. Default is None.
        name (str, optional): Name of the Weights & Biases run. Default is None.
        id (str, optional): ID of the Weights & Biases run. Default is None.
        tags (list, optional): List of tags for the Weights & Biases run. Default is None.
        group (str, optional): Group for the Weights & Biases run. Default is None.
    """

    dir: str = field(
        default_factory=lambda: str(Path(HydraConfig.get().runtime.output_dir))
    )
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None
    tags: Optional[list] = None
    group: Optional[str] = None


@dataclass
class OptimConfig:
    """Configuration for the optimization parameters.

    Args:
        optimizer (dict): Configuration for the optimizer.
        scheduler (dict): Configuration for the learning rate scheduler.
        epochs (int, optional): Number of epochs to train the model. Default is 1000.
        max_steps (int, optional): Maximum number of steps to train the model. Default is -1.
            If negative, the models trains on the full dataset.
            If it is between 0 and 1, it represents the fraction of the dataset to train on.
        accumulation_steps (int, optional): Number of steps to accumulate gradients before updating the model.
            Default is 1.
        grad_max_norm (float, optional): Maximum norm of the gradients. If None, no clipping is applied.
            Default is None.
    """

    optimizer: dict
    scheduler: dict
    epochs: int = 1000
    max_steps: int = -1
    accumulation_steps: int = 1
    grad_max_norm: Optional[float] = None
