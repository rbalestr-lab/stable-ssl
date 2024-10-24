from .joint_embedding.simclr import SimCLR
from .joint_embedding.barlow_twins import BarlowTwins
from .joint_embedding.wmse import WMSE
from .joint_embedding.vicreg import VICReg


__all__ = [
    "BaseModel",
    "Supervised",
    "SimCLR",
    "BarlowTwins",
    "VICReg",
    "WMSE",
]
