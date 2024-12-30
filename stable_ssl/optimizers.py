"""Optimizers."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """Implement LARS (Layer-wise Adaptive Rate Scaling) optimizer.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float
        Learning rate.
    momentum : float, optional
        Momentum factor. Default is 0.
    eta : float, optional
        LARS coefficient as used in the paper. Default is 1e-3.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default is 0.
    dampening : float, optional
        Dampening for momentum. Default is 0.
    nesterov : bool, optional
        Enables Nesterov momentum. Default is False.
    epsilon : float, optional
        Epsilon to prevent division by zero. Default is 0.

    """

    def __init__(
        self,
        params,
        lr=1e0,
        momentum=0,
        eta=1e-3,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        epsilon=0,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            eta=eta,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            epsilon=epsilon,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening."
            )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set the optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure: callable, optional
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-local_lr * group["lr"], d_p)

        return loss
