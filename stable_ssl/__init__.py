# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from .__about__ import (
    __title__,
    __summary__,
    __version__,
    __url__,
    __author__,
    __license__,
)

from .config import (
    get_args,
    # OptimConfig,
    # HardwareConfig,
    # LogConfig,
    # WandbConfig,
    # BaseModelConfig,
    # TrainerConfig,
)
from .data import DataConfig

from . import ssl_model
from .ssl_model import (
    SimCLR,
    BarlowTwins,
    VICReg,
    WMSE,
)

from . import utils

__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__url__",
    "__author__",
    "__license__",
    "get_args",
    "utils",
    # "OptimConfig",
    # "HardwareConfig",
    # "LogConfig",
    # "WandbConfig",
    # "BaseModelConfig",
    # "TrainerConfig",
    "DataConfig",
    "ssl_model",
    "SimCLR",
    "BarlowTwins",
    "VICReg",
    "WMSE",
]
