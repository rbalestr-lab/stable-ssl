import socket
from contextlib import contextmanager

import torch.distributed as dist


def find_free_port() -> int:
    """Find a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to any free port
        return s.getsockname()[1]


@contextmanager
def ddp_group_manager(rank, world_size, backend):
    """Context manager for DDP that gracefully closes process groups."""
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        yield
        dist.barrier()  # synchronize all workers at exit
    finally:
        dist.barrier()
        if rank == 0 and dist.is_initialized():
            dist.destroy_process_group()

        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
