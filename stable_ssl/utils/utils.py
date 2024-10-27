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


def setup_distributed(args):
    """Set up the distributed environment for PyTorch."""
    logging.info("Setting up Distributed model...")
    logging.info("exporting PyTorch distributed environment variables")
    dist_env = submitit.JobEnvironment()
    if "SLURM_JOB_NODELIST" in os.environ:
        cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
        host_name = subprocess.check_output(cmd).decode().splitlines()[0]
        dist_url = f"tcp://{host_name}:{args.port}"
    else:
        dist_url = f"tcp://localhost:{args.port}"
    logging.info(f"Process group:\n\t{dist_env.num_tasks} tasks")
    logging.info(f"\tmaster: {dist_url}")
    logging.info(f"\trank: {dist_env.global_rank}")
    logging.info(f"\tworld size: {dist_env.num_nodes*dist_env.num_tasks}")
    logging.info(f"\tlocal rank: {dist_env.local_rank}")
    # ToDo ?
    # os.environ["MASTER_ADDR"] = cluster_environment.main_address
    # os.environ["MASTER_PORT"] = str(cluster_environment.main_port)
    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available. Cannot initialize "
            "distributed process group."
        )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl",
            init_method=dist_url,
            rank=dist_env.global_rank,
            world_size=dist_env.num_nodes * dist_env.num_tasks,
        )
        args.world_size = dist_env.num_nodes * dist_env.num_tasks
        args.gpu = dist_env.local_rank
        assert dist_env.global_rank == torch.distributed.get_rank()
        assert (
            dist_env.num_nodes * dist_env.num_tasks
        ) == torch.distributed.get_world_size()
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
