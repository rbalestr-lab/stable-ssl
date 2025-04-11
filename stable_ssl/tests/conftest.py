import os

import pytest
import torch


@pytest.fixture
def config_file_path():
    """Return the absolute path to the `mini_mnist.yaml` file."""
    return os.path.join(os.path.dirname(__file__), "configs")


@pytest.fixture(scope="module", autouse=True)
def random_seed() -> None:
    """Module-level fixture to fix the random seed for reproducibility."""
    torch.manual_seed(1234)


@pytest.fixture
def device() -> torch.device:
    """Fixture to return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
