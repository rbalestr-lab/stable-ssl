# -*- coding: utf-8 -*-
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base import JointEmbeddingConfig, JointEmbeddingModel
from .barlow_twins import BarlowTwins, BarlowTwinsConfig
from .simclr import SimCLR, SimCLRConfig
from .vicreg import VICReg, VICRegConfig
from .wmse import WMSE, WMSEConfig
from .byol import BYOL, BYOLConfig

__all__ = [
    "JointEmbeddingConfig",
    "JointEmbeddingModel",
    "SimCLR",
    "SimCLRConfig",
    "BarlowTwins",
    "BarlowTwinsConfig",
    "VICReg",
    "VICRegConfig",
    "WMSE",
    "WMSEConfig",
    "BYOL",
    "BYOLConfig",
]
