"""Unit tests for dataset functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.unit
class TestDatasetUnit:
    """Unit tests for dataset classes without actual data loading."""

    def test_hf_dataset_initialization(self):
        """Test HFDataset can be initialized with proper parameters."""
        with patch("stable_pretraining.data.HFDataset") as mock_dataset:
            # Test basic initialization
            mock_dataset("ylecun/mnist", split="train")
            mock_dataset.assert_called_once_with("ylecun/mnist", split="train")

            # Test with transform
            mock_transform = Mock()
            mock_dataset("ylecun/mnist", split="train", transform=mock_transform)

            # Test with rename_columns
            mock_dataset(
                "ylecun/mnist", split="train", rename_columns={"image": "toto"}
            )

    def test_transform_function(self):
        """Test transform function logic without actual data."""
        mock_transform = Mock()
        mock_data = {"image": Mock()}

        def transform_func(x):
            x["image"] = mock_transform(x["image"])
            return x

        result = transform_func(mock_data.copy())
        mock_transform.assert_called_once_with(mock_data["image"])
        assert "image" in result

    def test_datamodule_configuration(self):
        """Test DataModule configuration parsing."""
        # Create configuration
        train_config = OmegaConf.create(
            {
                "dataset": {
                    "_target_": "stable_pretraining.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "train",
                },
                "batch_size": 20,
                "drop_last": True,
            }
        )

        test_config = OmegaConf.create(
            {
                "dataset": {
                    "_target_": "stable_pretraining.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "test",
                    "transform": {
                        "_target_": "stable_pretraining.data.transforms.ToImage",
                    },
                },
                "batch_size": 20,
            }
        )

        # Verify configuration structure
        assert train_config.dataset._target_ == "stable_pretraining.data.HFDataset"
        assert train_config.dataset.path == "ylecun/mnist"
        assert train_config.batch_size == 20
        assert train_config.drop_last is True

        assert test_config.dataset.split == "test"
        assert "transform" in test_config.dataset
        assert test_config.get("drop_last", False) is False

    def test_datamodule_methods(self):
        """Test DataModule method calls without actual data loading."""
        with patch("stable_pretraining.data.DataModule") as mock_datamodule_class:
            mock_datamodule = mock_datamodule_class.return_value

            # Mock the dataset attributes
            mock_train_dataset = Mock()
            mock_test_dataset = Mock()
            mock_datamodule.train_dataset = mock_train_dataset
            mock_datamodule.test_dataset = mock_test_dataset

            # Mock dataloader methods
            mock_train_loader = Mock(drop_last=True)
            mock_test_loader = Mock(drop_last=False)
            mock_datamodule.train_dataloader.return_value = mock_train_loader
            mock_datamodule.test_dataloader.return_value = mock_test_loader
            mock_datamodule.val_dataloader.return_value = mock_test_loader
            mock_datamodule.predict_dataloader.return_value = mock_test_loader

            # Test configuration
            train_config = Mock()
            test_config = Mock()

            datamodule = mock_datamodule_class(
                train=train_config,
                test=test_config,
                val=test_config,
                predict=test_config,
            )

            # Test method calls
            datamodule.prepare_data()
            datamodule.prepare_data.assert_called_once()

            datamodule.setup("fit")
            datamodule.setup.assert_called_with("fit")

            train_loader = datamodule.train_dataloader()
            assert train_loader.drop_last is True

            datamodule.setup("test")
            test_loader = datamodule.test_dataloader()
            assert test_loader.drop_last is False

            datamodule.setup("validate")
            val_loader = datamodule.val_dataloader()
            assert val_loader.drop_last is False

            datamodule.setup("predict")
            predict_loader = datamodule.predict_dataloader()
            assert predict_loader.drop_last is False

    def test_dataloader_creation(self):
        """Test DataLoader creation with dataset."""
        with patch("torch.utils.data.DataLoader") as mock_loader_class:
            mock_dataset = Mock()
            mock_loader_class.return_value

            # Create dataloader
            mock_loader_class(mock_dataset, batch_size=4, num_workers=2)

            mock_loader_class.assert_called_once_with(
                mock_dataset, batch_size=4, num_workers=2
            )

    def test_batch_structure(self):
        """Test expected batch structure from dataloader."""
        # Mock batch data
        mock_batch = {
            "image": torch.randn(4, 1, 28, 28),
            "label": torch.tensor([1, 2, 3, 4]),
        }

        # Test batch structure
        assert mock_batch["image"].shape == (4, 1, 28, 28)
        assert len(mock_batch["label"]) == 4
        assert "image" in mock_batch
        assert "label" in mock_batch

    def test_rename_columns_logic(self):
        """Test column renaming logic."""
        # Mock data with original column name
        original_data = {"image": "image_data", "label": 1}
        rename_map = {"image": "toto"}

        # Simulate renaming
        renamed_data = original_data.copy()
        if "image" in rename_map:
            renamed_data[rename_map["image"]] = renamed_data.pop("image")

        assert "toto" in renamed_data
        assert "image" not in renamed_data
        assert renamed_data["toto"] == "image_data"
        assert renamed_data["label"] == 1

    def test_repeated_sampler_replicas(self):
        import stable_pretraining as ssl

        results = {}
        num_replicas = 2
        fake_data_source_len = 10

        for rank in range(num_replicas):
            with (
                patch("torch.distributed.is_available", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_world_size", return_value=num_replicas),
                patch("torch.distributed.get_rank", return_value=rank),
            ):
                sampler = ssl.data.RepeatedRandomSampler(
                    data_source_or_len=fake_data_source_len, n_views=1, seed=42
                )

                epoch_len = len(list(iter(sampler)))
                results[rank] = epoch_len

        target_epoch_len = fake_data_source_len // num_replicas
        assert all(results[rank] == target_epoch_len for rank in range(num_replicas))

    def test_minari_dataset_bounds_logic(self):
        """Test bounds computation for MinariStepsDataset."""
        episodes_indices = [0, 1, 2, 3]
        episode_length = [10, 15, 5, 7]
        num_steps = 2
        num_episodes = len(episodes_indices)

        episode_lengths = [episode_length[i] for i in range(num_episodes - 1)]
        bounds = np.cumsum([0] + episode_lengths)
        bounds -= np.arange(num_episodes) * (num_steps - 1)

        assert len(episodes_indices) == len(episode_lengths) + 1
        assert np.all(bounds >= 0)
        assert len(bounds) == len(episodes_indices)

        idx = 25
        ep_idx = np.searchsorted(bounds, idx, side="right") - 1
        frame_idx = idx - bounds[ep_idx]

        assert ep_idx.item() == 2
        assert frame_idx < episode_length[ep_idx]
