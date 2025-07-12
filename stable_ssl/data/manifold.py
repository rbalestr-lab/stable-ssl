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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    xyz = swiss_roll(N=4000).numpy()
    color = xyz[:, 0] ** 2 + xyz[:, 2] ** 2
    color /= np.max(color)
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], cmap=plt.cm.jet, c=color)
    ax.set_title("Finite Data Samples", size=16)
    plt.tight_layout()
    plt.savefig("todelete.png")
    plt.close()

    cuts = torch.Tensor([0, 0.1, 0.8, 1])
    mix = torch.distributions.Categorical(torch.ones(len(cuts) - 1))
    comp = torch.distributions.Uniform(cuts[:-1], cuts[1:])
    p = torch.distributions.MixtureSameFamily(mix, comp)
    x = torch.linspace(0.2, 0.3, 100)
    print(p.sample())
    for xx in x:
        print(p.log_prob(x[[0]]))
    asfd
