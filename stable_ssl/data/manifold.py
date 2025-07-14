import numpy as np
import torch


def swiss_roll(
    N,
    margin=1,
    sampler_time=torch.distributions.uniform.Uniform(0.1, 3),
    sampler_width=torch.distributions.uniform.Uniform(0, 1),
):
    # draw samples to create the grid
    t0 = sampler_time.sample(sample_shape=(N,)) * 2 * np.pi
    radius = margin * t0 / np.pi + 0.1
    x = radius * torch.cos(t0)
    z = radius * torch.sin(t0)
    y = sampler_width.sample(sample_shape=(N,))
    xyz = torch.stack([x, y, z], 1)
    return xyz
