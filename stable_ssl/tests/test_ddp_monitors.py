import coverage
import pytest
import torch
import torch.multiprocessing as mp

from stable_ssl.monitors import LiDAR, RankMe
from stable_ssl.tests.utils import ddp_group_manager, find_free_port


def pytest_configure(config):
    """Configure coverage to handle multiprocessing."""
    coverage.process_startup()


@pytest.fixture
def ddp_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(find_free_port()))


def ddp_worker_lidar(rank: int, world_size: int, backend: str):
    """Worker function for testing LiDAR in DDP.

    Each spawned process calls this, with a unique rank.
    """
    with ddp_group_manager(rank, world_size, backend):
        local_n, q, d = 2, 2, 4
        local_embeddings = [torch.randn(q, d) + rank for _ in range(local_n)]
        lidar_monitor = LiDAR(n=world_size * local_n)

        score = lidar_monitor.lidar(local_embeddings)
        if rank == 0:
            assert isinstance(score, float), "LiDAR did not return a float in DDP test."


def ddp_worker_rankme(rank: int, world_size: int, backend: str):
    """Worker function for testing RankMe in DDP."""
    with ddp_group_manager(rank, world_size, backend):
        local_batch_size, d = 5, 8
        local_encoding = torch.randn(local_batch_size, d) + rank

        rankme_monitor = RankMe(limit=world_size * local_batch_size)

        score = rankme_monitor.rankme(local_encoding, epsilon=1e-7)

        if rank == 0:
            assert isinstance(
                score, float
            ), "RankMe did not return a float in DDP test."


@pytest.mark.usefixtures("ddp_env")
@pytest.mark.parametrize("backend", ["gloo"])
def test_lidar_ddp(backend: str):
    """Pytest test that spawns multiple CPU processes to test LiDAR in a DDP environment."""
    world_size = 2

    mp.spawn(ddp_worker_lidar, args=(world_size, backend), nprocs=world_size, join=True)


@pytest.mark.usefixtures("ddp_env")
@pytest.mark.parametrize("backend", ["gloo"])
def test_rankme_ddp(backend: str):
    """Pytest test that spawns multiple CPU processes to test RankMe in a DDP environment."""
    world_size = 2

    mp.spawn(
        ddp_worker_rankme, args=(world_size, backend), nprocs=world_size, join=True
    )
