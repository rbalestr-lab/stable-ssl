# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from .__about__ import (
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)
from .base import BaseTrainer
from .config import instanciate_config
from .losses import (
    BarlowTwinsLoss,
    NegativeCosineSimilarity,
    NNCLRLoss,
    NTXEntLoss,
    VICRegLoss,
)
from .modules import load_backbone
from .trainers import (
    JointEmbeddingPredictiveTrainer,
    JointEmbeddingTrainer,
    SelfDistillationTrainer,
    SupervisedTrainer,
)

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
    "JointEmbeddingPredictiveTrainer",
    "SelfDistillationTrainer",
    "NTXEntLoss",
    "NNCLRLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "NegativeCosineSimilarity",
    "instanciate_config",
    "random_mask",
]
