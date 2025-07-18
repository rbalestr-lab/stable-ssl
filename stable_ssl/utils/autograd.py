"""Custom autograd functions for SSL."""

import torch


class MyReLU(torch.autograd.Function):
    """Custom autograd Function for the Rectified Linear Unit (ReLU) activation.

    This Function clamps negative input values to zero while retaining positive values.
    The forward pass applies ReLU, and the backward pass propagates gradients
    only for inputs greater than zero.

    Args:
        ctx (torch.autograd.FunctionCtx):
            A context object provided by PyTorch's autograd engine.
            Use `ctx.save_for_backward(*tensors)` in `forward` to stash
            any tensors needed for gradient computation. In `backward`,
            retrieve those tensors via `ctx.saved_tensors` to compute
            gradients with respect to inputs.
        input (torch.Tensor):
            The input tensor to which ReLU is applied.

    Returns:
        torch.Tensor: The result of applying ReLU element-wise to the input.

    Example:
        >>> x = torch.tensor([-1.0, 2.0, -3.0])
        >>> MyReLU.apply(x)
        tensor([0.0, 2.0, 0.0])
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class OrderedCovariance(torch.autograd.Function):
    """Ordered covariance module.

    Computes a covariance loss that encourages ordering in the feature dimensions.
    During backpropagation, only the upper triangular part of the covariance
    matrix contributes to gradients.
    """

    @staticmethod
    def forward(ctx, X):
        C = (X.T @ X).fill_diagonal_(0)
        ctx.save_for_backward(X, C)
        return C.square().sum() / X.size(1)

    @staticmethod
    def backward(ctx, grad_output):
        X, C = ctx.saved_tensors
        # this would be the typical backprop:
        # X@C
        # instead we want to encourage ordering and thus use
        idx = torch.tril_indices(C.size(0), C.size(1))
        C[idx[0], idx[1]] = 0
        return 2 * X @ C * grad_output / X.size(1)


class Covariance(torch.autograd.Function):
    """Covariance module.

    Computes a standard covariance loss for decorrelation in self-supervised learning.
    The loss is the sum of squared off-diagonal elements of the covariance matrix.
    """

    @staticmethod
    def forward(ctx, X):
        C = (X.T @ X).fill_diagonal_(0)
        ctx.save_for_backward(X, C)
        return C.square().sum() / X.size(1)

    @staticmethod
    def backward(ctx, grad_output):
        X, C = ctx.saved_tensors
        return 4 * X @ C * grad_output / X.size(1)


# Convenience functions
ordered_covariance = OrderedCovariance.apply
covariance = Covariance.apply
