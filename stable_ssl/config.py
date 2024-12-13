# # -*- coding: utf-8 -*-
# """Configuration for stable-ssl runs."""
# #
# # Author: Hugues Van Assel <vanasselhugues@gmail.com>
# #         Randall Balestriero <randallbalestriero@gmail.com>
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

from typing import Optional, Union
from dataclasses import dataclass, field

# from omegaconf import OmegaConf
from pathlib import Path
import pickle
import lzma
import hydra


def instanciate_config(cfg=None, debug_hash=None) -> object:
    """Instanciate the config and debug hash."""
    if debug_hash is None:
        assert cfg is not None
        print("Your debugging hash:", lzma.compress(pickle.dumps(cfg)))
    else:
        print("Using debugging hash")
        cfg = pickle.loads(lzma.decompress(debug_hash))
    return hydra.utils.instantiate(cfg.trainer, _convert_="object", _recursive_=False)


@dataclass
class HardwareConfig:
    """Configuration for the 'hardware' parameters.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default is None.
    float16 : bool, optional
        Whether to use mixed precision (float16) for training. Default is False.
    gpu_id : int, optional
        GPU device ID to use for training. Default is 0.
    world_size : int, optional
        Number of processes participating in distributed training. Default is 1.
    port : int, optional
        Local proc's port number for distributed training. Default is None.
    """

    seed: Optional[int] = None
    float16: bool = False
    gpu_id: int = 0
    world_size: int = 1
    port: Optional[int] = None


@dataclass
class LoggerConfig:
    """Configuration for logging and checkpointing during training or evaluation.

    Parameters
    ----------
    level : int, optional
        The logging level. Determines the threshold for what gets logged. Default is 20.
    metrics : dict, optional
        A dictionary to store and log various metrics. Default is an empty dict.
    save_final_model : str, optional
        Specifies whether to save the final trained model.
        If a name is provided, the final model will be saved with that name.
        Default is "final".
    eval_every_epoch : int, optional
        The frequency (in epochs) at which the model will be evaluated.
        For example, if set to 1, evaluation occurs every epoch. Default is 1.
    every_step : int, optional
        The frequency (in training steps) at which to log intermediate metrics.
        For example, if set to 1, logs occur every step. Default is 1.
    checkpoint_frequency : int, optional
        The frequency (in epochs) at which model checkpoints are saved.
        For example, if set to 10, a checkpoint is saved every 10 epochs. Default is 10.
    checkpoint_model_only : bool, optional
        Whether to save only the model weights (True) or save additional training state
        (False) during checkpointing. Default is True.
    dump_path : pathlib.Path, optional
        The path where output is dumped. Defaults to Hydra's runtime output directory.
    wandb : bool or dict or None, optional
        Configuration for Weights & Biases logging.
        If `True`, it will be converted to an empty dictionary and default keys will be
        filled in if `rank == 0`. Default is None.
    """

    level: int = 20
    metrics: dict = field(default_factory=dict)
    save_final_model: str = "final"
    eval_every_epoch: int = 1
    every_step: int = 1
    checkpoint_frequency: int = 10
    checkpoint_model_only: bool = True
    dump_path: Path = field(
        default_factory=lambda: Path(HydraConfig.get().runtime.output_dir)
    )
    wandb: Union[bool, dict, None] = None


@dataclass
class WandbConfig(LogConfig):
    """Configuration for the Weights & Biases logging.

    Parameters
    ----------
    entity : str, optional
        Name of the (Weights & Biases) entity. Default is None.
    project : str, optional
        Name of the (Weights & Biases) project. Default is None.
    run : str, optional
        Name of the Weights & Biases run. Default is None.
    id : str, optional
        ID of the Weights & Biases run. Default is None.
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    run: Optional[str] = None
    id: Optional[str] = None
