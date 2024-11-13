# -*- coding: utf-8 -*-
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base import JointEmbedding
from .barlow_twins import BarlowTwinsLoss
from .simclr import NTXEnt
from .vicreg import VICReg
from .wmse import WMSE
from .byol import BYOL

__all__ = [
    "NTXEnt",
    "JointEmbedding",
    "BarlowTwinsLoss",
    "VICReg",
    "WMSE",
    "BYOL",
]
