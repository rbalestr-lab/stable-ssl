"""
This script demonstrates how to launch a run using the stable-SSL library.
"""

import hydra
import stable_ssl
import logging


@hydra.main()
def main(cfg):
    """Load the configuration and launch the run."""
    args = stable_ssl.get_args(cfg)  # Get the verified arguments

    logging.basicConfig(level=config.log.level, format="[stable-SSL] %(message)s")

    print("--- Arguments ---")
    print(args)

    # torch.autograd.set_detect_anomaly(True)

    stable_ssl.utils.get_gpu_info()
    model = getattr(stable_ssl, args.model.name)(args)  # Create model
    model()  # Call model


if __name__ == "__main__":
    main()
