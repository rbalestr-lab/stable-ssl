"""
This script demonstrates how to launch a run using the stable-SSL library.
python benchmarks/run.py
"""

import stable_ssl
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="config", config_name="default")
def main(cfg):
    """Load the configuration and launch the run."""
    trainer = stable_ssl.instanciate_config(cfg)
    trainer.setup()
    trainer.launch()
    print(trainer.get_logs())


if __name__ == "__main__":
    main()
