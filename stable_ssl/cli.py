# -*- coding: utf-8 -*-
"""Script to launch a stable-SSL run from the command line."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import hydra
from omegaconf import OmegaConf

# Register a resolver to evaluate expressions in the config file.
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2")
def main(cfg):
    """Load the configuration and launch stable-SSL run.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object loaded by Hydra.
    """
    trainer = hydra.utils.instantiate(
        cfg.trainer, _convert_="object", _recursive_=False
    )
    trainer()  # Call setup and launch methods.


def entry():
    """CLI entry point for the stable-ssl command."""
    import sys

    # We need to pass the config path as an absolute path to Hydra.
    if "--config-path" in sys.argv:
        index = sys.argv.index("--config-path")
        if index + 1 < len(sys.argv):
            config_path = sys.argv[index + 1]
            if not os.path.isabs(config_path):
                sys.argv[index + 1] = os.path.abspath(config_path)

    main()
