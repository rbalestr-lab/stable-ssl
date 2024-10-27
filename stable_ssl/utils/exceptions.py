# -*- coding: utf-8 -*-
"""Exceptions."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class BreakEpoch(Exception):
    pass


class BreakStep(Exception):
    pass


class NanError(Exception):
    pass


class BreakAllEpochs(Exception):
    pass
