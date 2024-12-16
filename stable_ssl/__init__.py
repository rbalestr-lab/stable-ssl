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

from .base import BaseModel
from .template import JointEmbedding, SelfDistillation
from .losses import NTXEntLoss, BYOLLoss, VICRegLoss, BarlowTwinsLoss
from .config import instanciate_config

__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__url__",
    "__author__",
    "__license__",
    "BaseModel",
    "JointEmbedding",
    "SelfDistillation",
    "NTXEntLoss",
    "BYOLLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "instanciate_config",
]
