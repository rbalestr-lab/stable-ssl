# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
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

from .base import BaseTrainer
from .trainers import SupervisedTrainer, JointEmbeddingTrainer, SelfDistillationTrainer
from .losses import NTXEntLoss, VICRegLoss, BarlowTwinsLoss, NegativeCosineSimilarity
from .config import instanciate_config
from .modules import load_backbone

__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__url__",
    "__author__",
    "__license__",
    "load_backbone",
    "BaseTrainer",
    "SupervisedTrainer",
    "JointEmbeddingTrainer",
    "SelfDistillationTrainer",
    "NTXEntLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "NegativeCosineSimilarity",
    "instanciate_config",
]
