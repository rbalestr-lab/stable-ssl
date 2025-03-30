"""Utility functions."""

#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import random
import socket
import subprocess
import time
from contextlib import closing
from functools import cache
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Module to gather tensors from all process. Supports backward propagation."""

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather(x: torch.Tensor):
    """All-gather tensors from all processes if DDP is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    else:
        return torch.cat(GatherLayer.apply(x), dim=0)


def gather(x: torch.Tensor, rank: int = 0):
    """Gathers a tensor to a specific rank."""
    if (
        not (dist.is_available() and dist.is_initialized())
        or (world_size := dist.get_world_size()) == 1
    ):
        return x

    if dist.get_rank() == rank:
        output = [torch.zeros_like(x) for _ in range(world_size)]
        dist.gather(x, output, dst=rank)
        return torch.cat(output, dim=0)
    else:
        dist.gather(x, [], dst=rank)
        return x


def all_reduce(x: torch.Tensor):
    """All-reduce tensors from all processes if DDP is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    else:
        x = x / dist.get_world_size()
        dist.all_reduce(x)
        return x


def reduce(x: torch.Tensor, rank: int = 0, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduces a tensor to a specific rank."""
    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        return x

    dist.reduce(x, dst=rank, op=op)
    return x


def broadcast(x: torch.Tensor, src_rank: int = 0):
    """Broadcasts a tensor from the specified rank to all devices."""
    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        return x

    # specified rank will have the correct x shape, so sync it across all processes
    shape = torch.tensor(x.shape, dtype=torch.int64, device=x.device)
    dist.broadcast(shape, src=src_rank)

    if dist.get_rank() != src_rank:
        x = torch.zeros(*shape.tolist(), dtype=x.dtype, device=x.device)

    dist.broadcast(x, src=src_rank)
    return x


@torch.no_grad()
def compute_global_mean(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    keepdim: bool = True,
    sync: bool = True,
) -> torch.Tensor:
    """Compute the mean of the input tensor across specified dimension(s).

    If running in a distributed environment (and `sync=True`), this function
    synchronizes the mean across all processes, effectively computing a global mean.
    """
    mean = torch.mean(x, dim=dim, keepdim=keepdim)

    if sync and dist.is_available() and dist.is_initialized():
        dist.all_reduce(mean, op=dist.ReduceOp.SUM)
        mean = mean / dist.get_world_size()

    return mean


def str_to_dtype(v: str) -> torch.dtype:
    """Convert a string to a pytorch dtype.

    Parameters
    ----------
    v: str
        the string value to infer as dtype

    Returns
    -------
        torch.dtype : torch.dtype
    """
    if v == "float32":
        return torch.float32
    if v == "float16":
        return torch.half
    if v == "float64":
        return torch.double


def count_SLURM_jobs(pending=True, running=True):
    """Count the number of SLURM jobs for the current user."""
    if pending and running:
        request = "pending,running"
    elif pending:
        request = "pending"
    else:
        request = "running"
    pipe = subprocess.Popen(
        ["squeue", "-u", os.environ["USER"], "-h", "-t", request, "-r"],
        stdout=subprocess.PIPE,
    )
    output = subprocess.check_output(("wc", "-l"), stdin=pipe.stdout)
    pipe.wait()
    return int(output)


def seed_everything(seed, fast=True):
    """Seed all random number generators."""
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def to_device(obj, device, non_blocking=True):
    """Recursively move tensors to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, tuple):
        return tuple(to_device(item, device, non_blocking) for item in obj)
    elif isinstance(obj, list):
        return [to_device(item, device, non_blocking) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    else:
        return obj


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, logging.error("Input tensor must be square.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_open_port():
    """Request the OS for any unused port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_local_rank():
    """Find the local rank of the current process.

    Find other procs with the same start command,
    sort them based on their pid and then assign ranks.
    Return the rank of the current process.
    """
    current_pid = os.getpid()
    cmd = f"ps -p {current_pid} -o command="
    current_command = subprocess.check_output(cmd, shell=True).decode().strip()

    cmd = f"ps -eo pid,command | grep '{current_command}' | grep -v grep"
    processes = subprocess.check_output(cmd, shell=True).decode().splitlines()
    processes = [p.split(None, 1) for p in processes]
    processes = [(int(p[0]), p[1:]) for p in processes]
    processes = sorted(processes, key=lambda x: x[0])

    rank = 0
    for p in processes:
        if p[0] == current_pid:
            break
        rank += 1
    return rank


def get_gpu_info():
    """Get the GPU device information using `nvidia-smi`.

    Torch information & CUDA_VISIBLE_DEVICES can be incomplete.
    """
    cmd = (
        "nvidia-smi --query-gpu="
        "name,memory.total,memory.used,pstate,pcie.link.gen.max,uuid,pci.bus_id "
        "--format=csv"
    )

    try:
        complete_process = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        logging.info("GPU info (nvidia-smi):")
        logging.info(f"\t{complete_process.stdout}")
    except subprocess.SubprocessError as e:
        logging.info("nvidia-smi failed.", exc_info=e)


def log_and_raise(exception_class, message):
    """Log an error message and raise an exception."""
    logging.error(message)
    raise exception_class(message)


@cache
def warn_once(*warnings: str):
    """Cache the warning message to avoid spamming the logs."""
    logging.warning("".join(warnings))


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
