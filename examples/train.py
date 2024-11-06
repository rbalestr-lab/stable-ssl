"""
This script demonstrates how to launch a run using the stable-SSL library.
"""

import logging
import hydra
import stable_ssl
from stable_ssl.utils import log_and_raise


@hydra.main()
def main(cfg):
    """Load the configuration and launch the run."""
    args = stable_ssl.get_args(cfg)  # Get the verified arguments

    logging.basicConfig(level=args.log.level, format="[stable-SSL] %(message)s")

    print("--- Arguments ---")
    print(args)

    # torch.autograd.set_detect_anomaly(True)

    stable_ssl.utils.get_gpu_info()

    # Check if the model name is a valid attribute in stable_ssl
    if hasattr(stable_ssl, args.model.name):
        model_class = getattr(stable_ssl, args.model.name)
        model = model_class(args)  # Create model instance
        model()  # Call model
    else:
        available_models = [
            attr for attr in dir(stable_ssl) if not attr.startswith("_")
        ]
        log_and_raise(
            ValueError,
            f"The model '{args.model.name}' is not available in stable_ssl. "
            f"Available models are: {available_models}.",
        )


if __name__ == "__main__":
    main()
