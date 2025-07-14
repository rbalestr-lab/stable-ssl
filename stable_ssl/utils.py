import functools
import inspect
from multiprocessing import Pool
from typing import Iterable, Optional, Union

import hydra
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import tqdm
from hydra.core.hydra_config import HydraConfig
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2


def get_required_fn_parameters(fn):
    sig = inspect.signature(fn)
    required = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


def dict_values(**kwargs):
    return list(kwargs.values())


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


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_gather(tensor, *args, **kwargs):
    if is_dist_avail_and_initialized():
        torch.distributed.nn.functional.all_gather(tensor, *args, **kwargs)
    return (tensor,)


def all_reduce(tensor, *args, **kwargs):
    if is_dist_avail_and_initialized():
        torch.distributed.nn.functional.all_reduce(tensor, *args, **kwargs)
    return tensor


class FullGatherLayer(torch.autograd.Function):
    """Gather tensors from all process and support backward propagation.

    Supports backward propagation for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if not torch.distributed.is_initialized():
            return x.unsqueeze(0)
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return torch.stack(output)

    @staticmethod
    def backward(ctx, grad):
        if not torch.distributed.is_initialized():
            return grad.squeeze(0)
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG)
        return grad[torch.distributed.get_rank()]


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
    """Ordered covariance module."""

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
    """Covariance module."""

    @staticmethod
    def forward(ctx, X):
        C = (X.T @ X).fill_diagonal_(0)
        ctx.save_for_backward(X, C)
        return C.square().sum() / X.size(1)

    @staticmethod
    def backward(ctx, grad_output):
        X, C = ctx.saved_tensors
        return 4 * X @ C * grad_output / X.size(1)


ordered_covariance = OrderedCovariance.apply
covariance = Covariance.apply


class Normalize(torch.nn.Module):
    """Normalize tensor and scale by square root of number of elements."""

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=(0, 1, 2)) * np.sqrt(x.numel())


class UnsortedQueue(torch.nn.Module):
    """A queue data structure that stores tensors with a maximum length."""

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
    """Exponential Moving Average module."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.item = torch.nn.UninitializedBuffer()

    def forward(self, item):
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


def generate_dae_samples(x, n, eps, num_workers=10):
    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    xtile = torch.repeat_interleave(x, n, dim=0)
    G = xtile.flatten(1).matmul(xtile.flatten(1).T)
    xtile.add_(torch.randn_like(xtile).mul_(torch.sqrt(torch.Tensor([eps]))))
    return xtile, G


def generate_sup_samples(x, y, n, num_workers=10):
    values, counts = np.unique(y, return_counts=True)

    values = values[counts >= n]
    values = np.flatnonzero(np.isin(y, values))
    ys = np.argsort(y[values])
    y = y[values[ys]]
    x = [x[i] for i in values[ys]]

    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    ytile = torch.nn.functional.one_hot(
        torch.from_numpy(y), num_classes=int(np.max(y) + 1)
    )
    G = ytile.flatten(1).matmul(ytile.flatten(1).T)
    return x, G


def generate_dm_samples(x, n, betas, i, num_workers=10):
    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    if not torch.is_tensor(betas):
        betas = torch.Tensor(betas)
    alphas = torch.cumprod(1 - betas, 0)
    xtile = torch.repeat_interleave(x, n * len(i), dim=0)
    alphas = torch.repeat_interleave(alphas[i], n).repeat(x.size(0))

    xtile.mul_(alphas.reshape(-1, 1, 1, 1).sqrt().expand_as(xtile))
    G = xtile.flatten(1).matmul(xtile.flatten(1).T)
    eps = (1 - alphas.reshape(-1, 1, 1, 1)).sqrt().expand_as(xtile)
    xtile.add_(torch.randn_like(xtile).mul_(eps))
    return xtile, G


def stable_eigvalsh(C: torch.Tensor, eps: float = None):
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    Id = torch.eye(C.size(1), device=C.device, dtype=C.dtype)
    C = C.div(rescalor).add(Id)

    eigvals = torch.linalg.eigvalsh(C)

    # - step 1: removing the effect of adding the Identity matrix
    # - step 2: removing the effect of dividing by P
    shift = torch.where(eigvals - 1 > eps, 1, eigvals - eps)
    return eigvals.sub(shift).mul(rescalor)


def stable_eigh(C, eps=None):
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    Id = torch.eye(C.size(1), device=C.device, dtype=C.dtype)
    C = C.div(rescalor).add(Id)

    eigvals, eigvecs = torch.linalg.eigh(C)

    # - step 1: removing the effect of adding the Identity matrix
    # - step 2: removing the effect of dividing by P
    shift = torch.where(eigvals - 1 > eps, 1, eigvals - eps)
    return eigvals.sub(shift).mul(rescalor), eigvecs


def stable_svd(C, eps=None, full_matrices=True):
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    C = C.div(rescalor)

    U, vals, Vh = torch.linalg.svd(C, full_matrices=full_matrices)

    # removing the effect of dividing by P
    return U, vals.clip(eps).mul(rescalor), Vh


def stable_svdvals(C, eps=None):
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    C = C.div(rescalor)
    vals = torch.linalg.svdvals(C)
    return vals.clip(eps).mul(rescalor)


def _apply_inet_transforms(x):
    transform = v2.Compose(
        [
            v2.RGB(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True, scale=(0.2, 0.99)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )
    return transform(x)


def generate_ssl_samples(x, n, num_workers=10):
    G = torch.kron(torch.eye(len(x)), torch.ones((n, n)))
    xtile = sum([[x[i] for _ in range(n)] for i in range(len(x))], [])
    with Pool(num_workers) as p:
        xtile = list(tqdm.tqdm(p.imap(_apply_inet_transforms, xtile), total=len(xtile)))
    return xtile, G


def _make_image(x):
    return (255 * (x - x.min()) / (x.max() - x.min())).int().permute(1, 2, 0)


def imshow_with_grid(
    ax,
    G: Union[np.ndarray, torch.Tensor],
    linewidth: Optional[float] = 0.4,
    color: Optional[Union[str, tuple]] = "black",
    bars=[],
    **kwargs,
):
    extent = [0, 1, 0, 1]
    if "extent" in kwargs:
        del kwargs["extent"]
    im = ax.imshow(G, extent=extent, **kwargs)
    shape = G.shape
    line_segments = []

    # horizontal lines
    for y in np.linspace(extent[2], extent[3], shape[1] + 1):
        line_segments.append([(extent[0], y), (extent[1], y)])
    # vertical lines
    for x in np.linspace(extent[0], extent[1], shape[0] + 1):
        line_segments.append([(x, extent[2]), (x, extent[3])])
    collection = LineCollection(line_segments, color=color, linewidth=linewidth)
    ax.add_collection(collection)

    # border line
    line_segments = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 1), (1, 1)],
        [(1, 0), (1, 1)],
    ]
    collection = LineCollection(line_segments, color="black", linewidth=linewidth * 3)
    ax.add_collection(collection)
    step = 1 / len(G)
    for bar in bars:
        if len(bar) == 2:
            barkwargs = {}
        else:
            barkwargs = bar[2]
        if "thickness" in barkwargs:
            thickness = barkwargs["thickness"]
            del barkwargs["thickness"]
        else:
            thickness = 1

        rect = Rectangle(
            xy=(bar[0] / len(G), 1),
            width=(bar[1] - bar[0]) / len(G),
            height=step * thickness,
            **barkwargs,
        )
        ax.add_patch(rect)
        rect = Rectangle(
            xy=(-step * thickness, 1 - bar[1] / len(G)),
            width=step * thickness,
            height=(bar[1] - bar[0]) / len(G),
            **barkwargs,
        )
        barkwargs["thickness"] = thickness
        ax.add_patch(rect)
        ax.set_xlim(-step * thickness, 1 + step)
        ax.set_ylim(-step, 1 + step * thickness)
    return im


def _plot_square(fig, x0, y0, x1, y1):
    fig.patches.append(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=None,
            edgecolor="tab:blue",
            linewidth=3,
            transform=fig.transFigure,
            figure=fig,
        )
    )


def visualize_images_graph(x, G, zoom_on=8):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_axis_off()

    # add in the overall graph
    inset = fig.add_axes([0.5, 0.005, 0.495, 0.8])
    imshow_with_grid(inset, G, lw=0.01)
    plt.setp(inset, xticks=[], yticks=[])
    bboxr = inset.get_position()

    # add in the zoomed one
    inset = fig.add_axes([0.04, 0.02, 0.42, 0.68])
    imshow_with_grid(inset, G[:zoom_on, :zoom_on], vmin=G.min(), vmax=G.max())
    plt.setp(inset, xticks=[], yticks=[])
    bboxl = inset.get_position()

    # add in the number of rows/columns
    dx = (np.max(bboxl.intervalx) - np.min(bboxl.intervalx)) / zoom_on
    dy = (np.max(bboxl.intervaly) - np.min(bboxl.intervaly)) / zoom_on
    for i in range(zoom_on):
        fig.text(
            np.min(bboxl.intervalx) + dx / 2 + dx * i,
            np.max(bboxl.intervaly) + dy / 3,
            str(i + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        fig.text(
            np.min(bboxl.intervalx) - dx / 3,
            np.max(bboxl.intervaly) - dy / 2 - dy * i,
            str(i + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    # add the inset zoom lines
    _plot_square(
        fig,
        np.min(bboxl.intervalx),
        np.min(bboxl.intervaly),
        np.max(bboxl.intervalx),
        np.max(bboxl.intervaly),
    )
    pct = zoom_on / G.size(0)
    delta_x = (np.max(bboxr.intervalx) - np.min(bboxr.intervalx)) * pct
    delta_y = (np.max(bboxr.intervaly) - np.min(bboxr.intervaly)) * pct
    _plot_square(
        fig,
        np.min(bboxr.intervalx),
        np.max(bboxr.intervaly) - delta_y,
        np.min(bboxr.intervalx) + delta_x,
        np.max(bboxr.intervaly),
    )
    fig.add_artist(
        lines.Line2D(
            [np.min(bboxl.intervalx), np.min(bboxr.intervalx)],
            [np.max(bboxl.intervaly), np.max(bboxr.intervaly)],
            linewidth=1,
        )
    )
    fig.add_artist(
        lines.Line2D(
            [np.max(bboxl.intervalx), np.min(bboxr.intervalx) + delta_x],
            [np.min(bboxl.intervaly), np.max(bboxr.intervaly) - delta_y],
            linewidth=1,
        )
    )

    # adding the images
    for i in range(zoom_on):
        inset = fig.add_axes(
            [0.002 + i / zoom_on, 0.815, 1 / (zoom_on + 1), 5.5 / 4 / zoom_on]
        )
        fig.text(
            i * 1 / zoom_on + 0.5 / zoom_on,
            1.003,
            rf"$x_{i + 1}$",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
        )
        inset.imshow(_make_image(x[i]), aspect="auto", interpolation="nearest")
        plt.setp(inset, xticks=[], yticks=[])

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)


def execute_from_config(manager, cfg):
    if "submitit" in cfg:
        assert "hydra" not in cfg
        hydra_conf = HydraConfig.get()
        # force_add ignores nodes in struct mode or Structured Configs nodes
        # and updates anyway, inserting keys as needed.
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            OmegaConf.update(
                cfg, "hydra.sweep.dir", hydra_conf.sweep.dir, force_add=True
            )
        with open_dict(cfg):
            cfg.hydra = {}
            cfg.hydra.job = OmegaConf.create(hydra_conf.job)
            cfg.hydra.sweep = OmegaConf.create(hydra_conf.sweep)
            cfg.hydra.run = OmegaConf.create(hydra_conf.run)

        executor = hydra.utils.instantiate(cfg.submitit.executor, _convert_="object")
        executor.update_parameters(**cfg.submitit.get("update_parameters", {}))
        job = executor.submit(manager)
        return job.result()
    else:
        return manager()


def adapt_resnet_for_lowres(model):
    model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    model.maxpool = torch.nn.Identity()
    return model


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    parent = rgetattr(obj, pre) if pre else obj
    if type(parent) is dict:
        parent[post] = val
    else:
        return setattr(parent, post, val)


def _adaptive_getattr(obj, attr):
    if type(obj) is dict:
        return obj[attr]
    else:
        return getattr(obj, attr)


def rgetattr(obj, attr):
    return functools.reduce(_adaptive_getattr, [obj] + attr.split("."))


def find_module(model: torch.nn.Module, module: torch.nn.Module):
    """Find modules in a model."""
    names = []
    values = []
    for child_name, child in model.named_modules():
        if isinstance(child, module):
            names.append(child_name)
            values.append(child)
    return names, values


def replace_module(model, replacement_mapping):
    """Replace a module in a model with another module."""
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input.")
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model
