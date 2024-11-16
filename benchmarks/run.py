"""
This script demonstrates how to launch a run using the stable-SSL library.
python benchmarks/run.py --config-dir benchmarks --config-name cifar10_simclr
"""

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2")
def main(cfg):
    """Load the configuration and launch the run."""
    conf = hydra.utils.instantiate(cfg, _convert_="object")
    conf["trainer"].execute()


if __name__ == "__main__":
    main()
