"""Exceptions."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class BreakEpoch(Exception):
    """Interrupt the current epoch."""

    pass


class BreakStep(Exception):
    """Interrupt the current training step."""

    pass


class NanError(Exception):
    """NaN error."""

    pass


class BreakAllEpochs(Exception):
    """Interrupt the training process across all epochs."""

    pass
