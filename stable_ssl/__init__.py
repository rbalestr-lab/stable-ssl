# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

import richuru

from . import backbone, callbacks, data, losses, module, optim, static, utils
from .__about__ import (
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)
from .manager import Manager
from .module import Module

__all__ = [
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
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
]
import sys

from loguru import logger

richuru.install()


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> (<cyan>{process}, {name}</cyan>) | <level>{message}</level>",
)


# # REDIRECT STANDARD PRINTING TO LOGURU
# class StreamToLoguru:
#     def __init__(self, logger, level="INFO"):
#         self.logger = logger
#         self.level = level

#     def write(self, message):
#         if message.strip():  # Avoid logging empty lines
#             self.logger.opt(depth=1).log(self.level, message.strip())

#     def flush(self):
#         pass  # Loguru handles flushing internally


# sys.stdout = StreamToLoguru(logger, "INFO")


# REDIRECT STANDARD LOGGING TO LOGURU
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

try:
    import datasets

    datasets.logging.set_verbosity_info()
except ModuleNotFoundError:
    pass
