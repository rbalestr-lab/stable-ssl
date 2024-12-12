# # -*- coding: utf-8 -*-
# """Configuration for stable-ssl runs."""
# #
# # Author: Hugues Van Assel <vanasselhugues@gmail.com>
# #         Randall Balestriero <randallbalestriero@gmail.com>
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging

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
class LogConfig:
    """Configuration for the 'log' parameters.

    Parameters
    ----------
    api: str, optional
        Which logging API to use.
        - Set to "wandb" to use Weights & Biases.
        - Set to "None" to use jsonlines.
        Default is None.
    folder : str, optional
        Path to the folder where logs and checkpoints will be saved.
        If None is provided, a default path is created under `./logs`.
        Default is None.
    load_from : str, optional
        Path to a checkpoint from which to load the model, optimizer, and scheduler.
        Default is "ckpt".
    level : int, optional
        Logging level (e.g., logging.INFO). Default is logging.INFO.
    checkpoint_frequency : int, optional
        Frequency of saving checkpoints (in terms of epochs). Default is 10.
    save_final_model : bool, optional
        Whether to save the final trained model. Default is True.
    final_model_name : str, optional
        Name for the final saved model. Default is "final_model".
    eval_only : bool, optional
        Whether to only evaluate the model without training. Default is False.
    eval_every_epoch : int, optional
        Frequency of evaluation (in terms of epochs). Default is 1.
    log_every_step : int, optional
        Frequency of logging (in terms of steps). Default is 1.
    """

    level: int = logging.INFO
    save_final_model: bool = True
    metrics: dict = None
    eval_every_epoch: int = 1
    log_every_step: int = 1
    wandb = False

    def __post_init__(self):
        """Initialize logging folder and run settings.

        If the folder path is not specified, creates a default path under `./logs`.
        The run identifier is set using the current timestamp if not provided.
        """
        if self.folder is None:
            self.folder = Path("./logs")
        else:
            self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)


# @dataclass
# class WandbConfig(LogConfig):
#     """Configuration for the Weights & Biases logging.

#     Parameters
#     ----------
#     entity : str, optional
#         Name of the (Weights & Biases) entity. Default is None.
#     project : str, optional
#         Name of the (Weights & Biases) project. Default is None.
#     run : str, optional
#         Name of the Weights & Biases run. Default is None.
#     rank_to_log: int, optional
#         Specifies the rank of the GPU/process to log for WandB tracking.
#         - Set to an integer value (e.g., 0, 1, 2) to log a specific GPU/process.
#         - Set to a negative value (e.g., -1) to log all processes.
#         Default is 0, which logs only the primary process.
#     """

#     entity: Optional[str] = None
#     project: Optional[str] = None
#     run: Optional[str] = None
#     rank_to_log: int = 0

#     def __post_init__(self):
#         """Check the rank to log for Weights & Biases."""
#         super().__post_init__()

#         if self.rank_to_log < 0:
#             raise ValueError("Cannot (yet) log all processes to Weights & Biases.")
