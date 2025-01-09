import pytest
import torch

from stable_ssl.monitors import LiDAR, RankMe


@pytest.fixture
def random_seed() -> None:
    """Pytest fixture to fix the random seed for reproducibility."""
    torch.manual_seed(1234)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# LIDAR TESTS
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("random_seed")
def test_lidar_smoke(device: torch.device) -> None:
    """Basic smoke test to ensure LiDAR runs without error and returns a float."""
    batch_size, q, d = 4, 2, 8
    embeddings = [torch.randn((q, d), device=device) for _ in range(batch_size)]

    lidar_monitor = LiDAR(n=8, epsilon=1e-7, delta=1e-3)

    score = lidar_monitor.lidar(embeddings)

    assert isinstance(score, float), "LiDAR did not return a float."
    assert score > 0, f"LiDAR returned non-positive value: {score}"


@pytest.mark.usefixtures("random_seed")
def test_lidar_identical_embeddings(device: torch.device) -> None:
    """Test LiDAR with identical embeddings.

    If all embeddings are identical, the between-class scatter should be zero,
    often leading LiDAR to a trivial value
    (usually ~1, but numeric issues might shift it).
    """
    batch_size, q, d = 4, 2, 8
    identical_vec = torch.ones((q, d), device=device)
    embeddings = [identical_vec for _ in range(batch_size)]

    lidar_monitor = LiDAR(n=batch_size)
    score = lidar_monitor.lidar(embeddings)

    assert 0.0 < score < 2.0, f"Unexpected LiDAR for identical embeddings: {score}"


@pytest.mark.usefixtures("random_seed")
def test_lidar_single_class(device: torch.device) -> None:
    """Test LiDAR with a single class.

    Edge case: single class (batch_size=1).
    May produce (n-1)=0 in denominator. We can check if the code handles it gracefully.
    """
    batch_size, q, d = 1, 2, 8
    embeddings = [torch.randn((q, d), device=device) for _ in range(batch_size)]

    lidar_monitor = LiDAR(n=1)
    score = lidar_monitor.lidar(embeddings)

    assert isinstance(
        score, float
    ), "LiDAR did not produce a float with single-class input."
    assert score > 0, "LiDAR is expected to be > 0 even for single-class edge case."


# --------------------------------------------------------------------------
# RANKME TESTS
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("random_seed")
def test_rankme_smoke(device: torch.device) -> None:
    """Basic smoke test for RankMe to ensure it runs and returns a float."""
    batch_size, d = 16, 32
    encoding = torch.randn((batch_size, d), device=device)

    rankme_monitor = RankMe(limit=16, epsilon=1e-7)
    score = rankme_monitor.rankme(encoding, epsilon=1e-7)

    assert isinstance(score, float), "RankMe did not return a float."
    assert score > 0, f"RankMe returned a non-positive value: {score}"


@pytest.mark.usefixtures("random_seed")
def test_rankme_identical_embeddings(device: torch.device) -> None:
    """Test RankMe with identical embeddings.

    If all embeddings are identical, the singular values should be mostly zero except
    for one direction => RankMe often yields ~1.
    """
    batch_size, d = 8, 16
    identical_encoding = torch.ones((batch_size, d))

    rankme_monitor = RankMe(limit=batch_size)
    score = rankme_monitor.rankme(identical_encoding, epsilon=1e-7)

    # If everything is identical, effectively there's rank=1 =>
    # we expect an "effective rank" near 1. We can check it's around [0.5, 1.5].
    assert 0.0 < score < 2.0, f"RankMe unexpected for identical embeddings: {score}"


@pytest.mark.usefixtures("random_seed")
def test_rankme_multi_view(device: torch.device) -> None:
    """Tests feeding a list of encodings for multiple views."""
    batch_size, d = 8, 32
    view1 = torch.randn((batch_size, d), device=device)
    view2 = torch.randn((batch_size, d), device=device)

    rankme_monitor = RankMe(limit=16)
    score1 = rankme_monitor.rankme(view1, rankme_monitor.epsilon)
    score2 = rankme_monitor.rankme(view2, rankme_monitor.epsilon)

    assert isinstance(score1, float)
    assert isinstance(score2, float)
    assert score1 > 0 and score2 > 0
