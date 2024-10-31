# -*- coding: utf-8 -*-
"""Utility functions."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
import numpy as np
import subprocess
from time import time
import logging
import torch.distributed as dist
import submitit
import torch


class FullGatherLayer(torch.autograd.Function):
    """Gather tensors from all process. Supports backward propagation."""

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


def setup_distributed(args, launcher="submitit_local"):
    """Set up the distributed environment for PyTorch."""
    logging.info("Setting up Distributed model...")
    logging.info("exporting PyTorch distributed environment variables")
    dist_env = None
    if launcher is not None and "submitit" in launcher:
        # hydra's laucher pluging being used
        dist_env = submitit.JobEnvironment()
        world_size = dist_env.num_nodes * dist_env.num_tasks
    if "SLURM_JOB_NODELIST" in os.environ:
        # slurm manager being used irrespective of hydra
        cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
        host_name = subprocess.check_output(cmd).decode().splitlines()[0]
        dist_url = f"tcp://{host_name}:{args.port}"
        if dist_env is None:
            dist_env = {
                "num_tasks": int(os.getenv("SLURM_NTASKS", 1)),
                "global_rank": int(os.getenv("SLURM_PROCID", 0)),
                "local_rank": int(os.getenv("SLURM_LOCALID", 0)),
            }
            world_size = dist_env.get("num_tasks", 1)
    else:
        # local host being used irrespective of hydra
        host_name = "localhost"
        dist_url = f"tcp://{host_name}:{args.port}"
        if dist_env is None:
            cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
            num_gpus = subprocess.check_output(cmd, shell=True).decode().splitlines()[0]
            rank = 0  # hardcoding rank to 0 for local host
            dist_env = {
                "num_tasks": int(num_gpus),
                "global_rank": rank,
                "local_rank": rank,
            }
            world_size = dist_env.get("num_tasks", 1)
    logging.info(f"Process group:\n\t{dist_env.get('num_tasks', 1)} tasks")
    logging.info(f"\tmaster: {dist_url}")
    logging.info(f"\trank: {dist_env.get('global_rank', 0)}")
    logging.info(f"\tworld size: {world_size}")
    logging.info(f"\tlocal rank: {dist_env.get('local_rank', 0)}")

    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = args.port
    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available. Cannot initialize "
            "distributed process group."
        )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl",
            init_method=dist_url,
            rank=dist_env.get("global_rank", 0),
            world_size=world_size,
        )
        args.world_size = world_size
        args.gpu = dist_env.get("local_rank", 0)
        assert dist_env.get("global_rank", 0) == torch.distributed.get_rank()
        assert (world_size) == torch.distributed.get_world_size()
    return args


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
        seed = int(time())
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
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
