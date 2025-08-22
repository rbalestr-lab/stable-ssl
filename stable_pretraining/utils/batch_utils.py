"""Utility functions for handling batch and outputs dictionaries in callbacks."""

from typing import Any, Dict, Optional

from loguru import logger as logging


def get_data_from_batch_or_outputs(
    key: str,
    batch: Dict[str, Any],
    outputs: Optional[Dict[str, Any]] = None,
    caller_name: str = "Callback",
) -> Optional[Any]:
    """Get data from either outputs or batch dictionary.

    In PyTorch Lightning, the outputs parameter in callbacks contains the return
    value from training_step/validation_step, while batch contains the original
    input. Since forward methods may modify batch in-place but Lightning creates
    a copy for outputs, we need to check both.

    Args:
        key: The key to look for in the dictionaries
        batch: The original batch dictionary
        outputs: The outputs dictionary from training/validation step
        caller_name: Name of the calling function/class for logging

    Returns:
        The data associated with the key, or None if not found
    """
    # First check outputs (which contains the forward pass results)
    if outputs is not None and key in outputs:
        return outputs[key]
    elif key in batch:
        return batch[key]
    else:
        logging.warning(
            f"{caller_name}: Key '{key}' not found in batch or outputs. "
            f"Available batch keys: {list(batch.keys())}, "
            f"Available output keys: {list(outputs.keys()) if outputs else 'None'}"
        )
        return None
