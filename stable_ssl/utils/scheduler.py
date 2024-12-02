# -*- coding: utf-8 -*-
"""Learning rate schedulers."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import (
    LinearLR,
    MultiStepLR,
    CosineAnnealingLR,
    SequentialLR,
    LambdaLR,
)
import numpy as np


class CosineDecayer:
    """Apply cosine decay with multiple cycles for learning rate scheduling."""

    def __init__(self, total_steps, n_cycles=3, gamma=0.2):
        self.total_steps = total_steps
        self.n_cycles = n_cycles

    def __call__(self, step):
        alpha = 1 - step / self.total_steps
        cycle = 1 + np.sin(self.n_cycles * 2 * np.pi * step / self.total_steps) / 2
        return alpha * cycle


def LinearWarmup(optimizer, total_steps, start_factor=0.01, peak_step=0.1):
    """Create a linear warmup scheduler."""
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    return warmup


def LinearWarmupCosineAnnealing(
    optimizer, total_steps, start_factor=0.01, end_lr=0.0001, peak_step=0.01
):
    """Combine linear warmup with cosine annealing decay."""
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = CosineAnnealingLR(optimizer, T_max=total_steps - peak_step, eta_min=end_lr)
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler


def LinearWarmupCyclicAnnealing(
    optimizer, total_steps, start_factor=0.01, peak_step=0.1
):
    """Combine linear warmup with cyclic cosine annealing."""
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)

    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    decay = LambdaLR(optimizer, CosineDecayer(total_steps - peak_step))
    scheduler = SequentialLR(
        optimizer,
        [warmup, decay],
        milestones=[peak_step],
    )
    return scheduler


def LinearWarmupThreeStepsAnnealing(
    optimizer, total_steps, start_factor=0.001, gamma=0.3, peak_step=0.05
):
    """Combine linear warmup with a three-step learning rate annealing."""
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = MultiStepLR(
        optimizer,
        milestones=[
            (total_steps - peak_step) * 0.4,
            (total_steps - peak_step) * 0.6,
            (total_steps - peak_step) * 0.8,
        ],
        gamma=gamma,
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    warmup = []
    cosine_annealing = []
    threestep_annealing = []
    cyclic = []

    W = torch.nn.Parameter(torch.tensor([0.0]))
    for data, option in zip(
        [warmup, cosine_annealing, threestep_annealing, cyclic],
        [
            LinearWarmup,
            LinearWarmupCosineAnnealing,
            LinearWarmupThreeStepsAnnealing,
            LinearWarmupCyclicAnnealing,
        ],
    ):
        optim = torch.optim.SGD([W], lr=1)
        sched = option(optim, 100)
        data.append(sched.get_last_lr())
        for i in range(100):
            loss = W.sum()
            optim.step()
            sched.step()
            data.append(sched.get_last_lr())
        print("-------------")

    plt.plot(warmup)
    plt.plot(cosine_annealing)
    plt.plot(threestep_annealing)
    plt.plot(cyclic)
    plt.show()
