"""Neural network modules and utilities."""

from typing import Iterable, Union

import numpy as np
import torch
import torch.nn.functional as F


class ImageToVideoEncoder(torch.nn.Module):
    """Wrapper to apply an image encoder to video data by processing each frame independently.

    This module takes video data with shape (batch, time, channel, height, width) and applies
    an image encoder to each frame, returning the encoded features.

    Args:
        encoder (torch.nn.Module): The image encoder module to apply to each frame.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        # we expect something of the shape
        # (batch, time, channel, height, width)
        batch_size, num_timesteps = video.shape[:2]
        assert video.ndim == 5
        # (BxT)xCxHxW
        video = video.contiguous().flatten(0, 1)
        # (BxT)xF
        features = self.encoder(video)
        # BxTxF
        features = features.contiguous().view(
            batch_size, num_timesteps, features.size(1)
        )
        return features


class Normalize(torch.nn.Module):
    """Normalize tensor and scale by square root of number of elements."""

    def forward(self, x):
        return F.normalize(x, dim=(0, 1, 2)) * np.sqrt(x.numel())


class UnsortedQueue(torch.nn.Module):
    """A queue data structure that stores tensors with a maximum length.

    This module implements a circular buffer that stores tensors up to a
    maximum length. When the queue is full, new items overwrite the oldest ones.

    Args:
        max_length: Maximum number of elements to store in the queue
        shape: Shape of each element (excluding batch dimension). Can be int or tuple
        dtype: Data type of the tensors to store
    """

    def __init__(
        self, max_length: int, shape: Union[int, Iterable[int]] = None, dtype=None
    ):
        super().__init__()
        self.max_length = max_length
        self.pointer = torch.nn.Buffer(torch.zeros((), dtype=torch.long))
        self.filled = torch.nn.Buffer(torch.zeros((), dtype=torch.bool))
        if shape is None:
            self.out = torch.nn.UninitializedBuffer()
        else:
            if type(shape) is int:
                shape = (shape,)
            self.out = torch.nn.Buffer(
                torch.zeros((max_length,) + tuple(shape), dtype=dtype)
            )

    def append(self, item):
        """Append item(s) to the queue.

        Args:
            item: Tensor to append. First dimension is batch size.

        Returns:
            Current contents of the queue
        """
        if self.max_length == 0:
            return item
        if isinstance(self.out, torch.nn.parameter.UninitializedBuffer):
            shape = (self.max_length,) + item.shape[1:]
            self.out.materialize(shape, dtype=item.dtype, device=item.device)
            torch.nn.init.zeros_(self.out)
        if self.pointer + item.size(0) < self.max_length:
            self.out[self.pointer : self.pointer + item.size(0)] = item
            self.pointer.add_(item.size(0))
        else:
            remaining = self.max_length - self.pointer
            self.out[-remaining:] = item[:remaining]
            self.out[: item.size(0) - remaining] = item[remaining:]
            self.pointer.copy_(item.size(0) - remaining)
            self.filled.copy_(True)
        return self.out if self.filled else self.out[: self.pointer]

    def get(self):
        """Get current contents of the queue.

        Returns:
            Tensor containing all items currently in the queue
        """
        return self.out if self.filled else self.out[: self.pointer]

    @staticmethod
    def _test():
        q = UnsortedQueue(0)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            assert v.numel() == 1
            assert v[0] == i

        q = UnsortedQueue(5)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            if i < 5:
                assert v[-1] == i
        assert v.numel() == 5
        assert 9 in v.numpy()
        assert 8 in v.numpy()
        assert 7 in v.numpy()
        assert 6 in v.numpy()
        assert 5 in v.numpy()
        assert 4 not in v.numpy()
        assert 3 not in v.numpy()
        assert 2 not in v.numpy()
        assert 1 not in v.numpy()
        assert 0 not in v.numpy()
        return True


class EMA(torch.nn.Module):
    """Exponential Moving Average module.

    Maintains an exponential moving average of input tensors.

    Args:
        alpha: Smoothing factor between 0 and 1.
               0 = no update (always return first value)
               1 = no smoothing (always return current value)
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.item = torch.nn.UninitializedBuffer()

    def forward(self, item):
        """Update EMA and return smoothed value.

        Args:
            item: New tensor to incorporate into the average

        Returns:
            Exponentially smoothed tensor
        """
        if self.alpha < 1 and isinstance(self.item, torch.nn.UninitializedBuffer):
            with torch.no_grad():
                self.item.materialize(
                    shape=item.shape, dtype=item.dtype, device=item.device
                )
                self.item.copy_(item, non_blocking=True)
            return item
        elif self.alpha == 1:
            return item
        with torch.no_grad():
            self.item.mul_(1 - self.alpha)
        output = item.mul(self.alpha).add(self.item)
        with torch.no_grad():
            self.item.copy_(output)
        return output

    @staticmethod
    def _test():
        q = EMA(0)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            v = q(torch.randn(10, 10))
            assert torch.allclose(v, R)
        q = EMA(1)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            assert torch.allclose(v, R)

        q = EMA(0.5)
        R = torch.randn(10, 10)
        ground = R.detach()
        v = q(R)
        assert torch.allclose(ground, v)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            ground = R * 0.5 + ground * 0.5
            assert torch.allclose(v, ground)
        return True
