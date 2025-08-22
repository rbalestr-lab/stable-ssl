"""Unit tests for data samplers."""

import pytest
import torch
from torch.utils.data import Dataset

import stable_pretraining as spt


class MockDataset(Dataset):
    """Simple mock dataset for testing samplers."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"index": idx, "data": torch.randn(3)}


@pytest.mark.unit
class TestSamplers:
    """Test sampler functionality without actual data loading."""

    @pytest.mark.parametrize("n_views", [1, 2, 4])
    def test_repeated_random_sampler_indices(self, n_views):
        """Test RepeatedRandomSampler generates correct indices."""
        dataset_size = 10
        sampler = spt.data.sampler.RepeatedRandomSampler(
            data_source_or_len=dataset_size, n_views=n_views
        )

        # Collect indices from one epoch
        indices = list(sampler)

        # Should have dataset_size * n_views indices
        assert len(indices) == dataset_size * n_views

        # Each original index should appear n_views times
        index_counts = {}
        for idx in indices:
            index_counts[idx] = index_counts.get(idx, 0) + 1

        for count in index_counts.values():
            assert count == n_views

    def test_repeated_random_sampler_with_dataset(self):
        """Test RepeatedRandomSampler with actual dataset."""
        dataset = MockDataset(size=20)
        sampler = spt.data.sampler.RepeatedRandomSampler(
            data_source_or_len=dataset, n_views=2
        )

        indices = list(sampler)
        assert len(indices) == 40  # 20 * 2

    def test_random_subset_sampler(self):
        """Test RandomSubsetSampler if it exists."""
        if hasattr(spt.data.sampler, "RandomSubsetSampler"):
            dataset_size = 100
            subset_size = 20

            sampler = spt.data.sampler.RandomSubsetSampler(
                data_source_or_len=dataset_size, subset_size=subset_size
            )

            indices = list(sampler)
            assert len(indices) == subset_size
            assert all(0 <= idx < dataset_size for idx in indices)
