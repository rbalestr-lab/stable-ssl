# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

from . import backbone, callbacks, data, losses, module, optim, static, utils
from .__about__ import (
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)
from .backbone.utils import TeacherStudentWrapper
from .callbacks import (
    EarlyStopping,
    ImageRetrieval,
    LiDAR,
    LoggingCallback,
    ModuleSummary,
    OnlineKNN,
    OnlineProbe,
    OnlineWriter,
    RankMe,
    SklearnCheckpoint,
    TeacherStudentCallback,
    TrainerInfo,
)
from .manager import Manager
from .module import Module

__all__ = [
    OnlineProbe,
    SklearnCheckpoint,
    OnlineKNN,
    TrainerInfo,
    LoggingCallback,
    ModuleSummary,
    EarlyStopping,
    OnlineWriter,
    RankMe,
    LiDAR,
    ImageRetrieval,
    TeacherStudentCallback,
    utils,
    data,
    module,
    static,
    utils,
    optim,
    losses,
    callbacks,
    Manager,
    backbone,
    Module,
    TeacherStudentWrapper,
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
]

# Setup logging
from loguru import logger

# Try to install richuru for better formatting if available
try:
    import richuru

    richuru.install()
except ImportError:
    pass


def rank_zero_only_filter(record):
    """Filter to only log on rank 0 in distributed training."""
    import os

    # Check common environment variables for distributed rank
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return rank == "0"


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> (<cyan>{process}, {name}</cyan>) | <level>{message}</level>",
    filter=rank_zero_only_filter,
)


# Redirect standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Find caller from where originated the log message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Remove all handlers associated with the root logger object
logging.root.handlers = []
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Try to set datasets logging verbosity if available
try:
    import datasets

    datasets.logging.set_verbosity_info()
except ModuleNotFoundError:
    pass
