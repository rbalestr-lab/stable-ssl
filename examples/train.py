"""
This script demonstrates how to launch a run using the stable-SSL library.
python examples/train.py --config-dir examples/configs/simclr --config-name cifar10_requeue
"""

import hydra


@hydra.main(version_base="1.2")
def main(cfg):
    """Load the configuration and launch the run."""
    trainer = hydra.utils.instantiate(
        cfg.trainer, _convert_="object", _recursive_=False
    )
    trainer.setup()
    trainer.launch()


if __name__ == "__main__":
    main()
