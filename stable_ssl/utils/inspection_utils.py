"""Function inspection and general helper utilities."""

import inspect
from typing import Any, List


def get_required_fn_parameters(fn):
    """Get the list of required parameters for a function.

    Args:
        fn: The function to inspect

    Returns:
        List of parameter names that don't have default values
    """
    sig = inspect.signature(fn)
    required = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


def dict_values(**kwargs):
    """Convert keyword arguments to a list of values.

    Returns:
        List of values from the provided keyword arguments
    """
    return list(kwargs.values())


def broadcast_param_to_list(
    param: Any, target_length: int, param_name: str
) -> List[Any]:
    """Broadcast a parameter value to create a list of specified length.

    This function handles the common pattern of accepting either:
    - None: creates a list of None values
    - A single value: broadcasts to all positions
    - A single-element list/tuple: broadcasts the element to all positions
    - A list/tuple of correct length: returns as-is

    Args:
        param: The parameter to broadcast (can be None, single value, or list/tuple)
        target_length: The desired length of the output list
        param_name: Name of the parameter for error messages

    Returns:
        List of values with length matching target_length

    Raises:
        ValueError: If param is a list/tuple with length > 1 that doesn't match target_length

    Examples:
        >>> broadcast_param_to_list(None, 3, "dims")
        [None, None, None]
        >>> broadcast_param_to_list(5, 3, "dims")
        [5, 5, 5]
        >>> broadcast_param_to_list([5], 3, "dims")
        [5, 5, 5]
        >>> broadcast_param_to_list([1, 2, 3], 3, "dims")
        [1, 2, 3]
    """
    if param is None:
        return [None] * target_length

    if not isinstance(param, (list, tuple)):
        # Single value provided for all elements
        return [param] * target_length

    if len(param) == 1 and target_length > 1:
        # Single value in list, broadcast to all elements
        return list(param) * target_length

    if len(param) != target_length:
        raise ValueError(
            f"Length of {param_name} ({len(param)}) must match target length ({target_length})"
        )

    return list(param)
