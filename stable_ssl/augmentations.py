import inspect
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import zoom as scizoom
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from stable_ssl.utils import log_and_raise


@dataclass
class TransformsConfig:
    """Configuration for the data used for training the model.

    Parameters
    ----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str = None
    transforms: list[dict] = None

    def __post_init__(self):
        """Initialize the transformation configuration."""
        extra = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.transforms is None:
            self.transforms = [{}]
            self._transform = v2.Compose(extra)
        else:
            self._transform = v2.Compose(
                [TransformConfig(**t) for t in self.transforms] + extra
            )

    def __call__(self, x):
        return self._transform(x)


def get_interpolation_mode(mode_str: str) -> InterpolationMode:
    """Get the interpolation mode from a string."""
    try:
        return InterpolationMode(mode_str)
    except ValueError:
        log_and_raise(
            ValueError,
            f"{mode_str} is not a valid interpolation mode. "
            f"Choose from {list(InterpolationMode)}.",
        )


@dataclass
class TransformConfig:
    """Configuration for the data used for training the model.

    Parameters
    ----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str
    args: list
    kwargs: dict
    p: float = 1.0

    def __init__(self, name, args=None, kwargs=None, p=1):
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}

        if "interpolation" in self.kwargs:
            self.kwargs["interpolation"] = get_interpolation_mode(
                self.kwargs["interpolation"]
            )

        self.p = p
        if self.name is not None:
            if self.name in globals():
                func = globals()[self.name]
            else:
                func = getattr(v2, self.name, None)
                if func is None:
                    log_and_raise(
                        AttributeError,
                        f"'{self.name}' not found in globals() or in 'v2'. "
                        "Please check the function name.",
                    )

            # Check if the function has a p argument.
            func_signature = inspect.signature(func)
            p_in_args = "p" in func_signature.parameters

            t = func(*self.args, **self.kwargs)

            if self.p < 1:
                if p_in_args:
                    log_and_raise(
                        ValueError,
                        f"The function '{self.name}' already includes a 'p' argument, "
                        f"but p={self.p} is also set externally in the configuration. "
                        "This results in 'p' being applied twice. "
                        "Please adjust the configuration to set 'p' only via the "
                        f"kwargs of the function '{self.name}'.",
                    )
                self._transform = v2.RandomApply(torch.nn.ModuleList([t]), p=self.p)
            elif self.p == 0:
                self._transform = v2.Identity()
            else:
                self._transform = t
        else:
            self._transform = v2.Identity()

    def __call__(self, x):
        return self._transform(x)

class GaussianBlur(torch.nn.Module):
    """Apply Gaussian blur to an image.

    Unlike the torchvision implementation, this one does not require the kernel size.
    """

    def __init__(
        self,
        kernel_size: Optional[float] = None,
        sigma: Tuple[float, float] = (0.1, 2),
    ):
        super().__init__()
        if kernel_size is not None:
            logging.warning(
                "The 'kernel_size' argument of the GaussianBlur "
                "augmentation will be deprecated. "
            )
        self.sigma = sigma

    def forward(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))
