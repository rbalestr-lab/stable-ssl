"""
This script demonstrates how to launch a run using the stable-SSL library.
"""

import logging
import hydra
import stable_ssl
from stable_ssl.utils import log_and_raise

__all__ = ["log_and_raise", "stable_ssl", "logging"]


@hydra.main(version_base="1.2")
def main(cfg):
    """Load the configuration and launch the run."""
    conf = hydra.utils.instantiate(cfg, _convert_="object")
    conf["trainer"].execute()


if __name__ == "__main__":
    main()
