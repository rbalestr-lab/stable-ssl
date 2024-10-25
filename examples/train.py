"""
This script demonstrates how to train a model using the stable-SSL library.
"""

import hydra
from omegaconf import DictConfig

import stable_ssl

from stable_ssl import SimCLR, Supervised

model_dict = {
    "SimCLR": SimCLR,
    "Supervised": Supervised,
}


@hydra.main()
def main(cfg: DictConfig):
    args = stable_ssl.get_args(cfg)

    print("--- Arguments ---")
    print(args)

    trainer = model_dict[args.model.name](args)
    trainer()


if __name__ == "__main__":
    main()
