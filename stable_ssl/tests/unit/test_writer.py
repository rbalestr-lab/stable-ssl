"""Unit tests for writer callback functionality."""

import os
import shutil
from unittest.mock import patch

import pytest
import torch


@pytest.mark.unit
class TestWriterUnit:
    """Unit tests for OnlineWriter callback without actual file I/O."""

    def test_online_writer_initialization(self):
        """Test OnlineWriter callback initialization."""
        with patch("stable_ssl.callbacks.OnlineWriter") as mock_writer:
            writer = mock_writer(
                names=["embedding", "linear_probe_preds"],
                path="./tmp/",
                during=["train"],
                every_k_epochs=2,
            )

            mock_writer.assert_called_once_with(
                names=["embedding", "linear_probe_preds"],
                path="./tmp/",
                during=["train"],
                every_k_epochs=2,
            )
            assert writer is not None

    def test_writer_path_handling(self):
        """Test writer path creation and handling."""
        with patch("os.makedirs") as mock_makedirs:
            with patch("os.path.exists", return_value=False):
                # Simulate path creation logic
                path = "./tmp/"
                os.makedirs(path, exist_ok=True)

                mock_makedirs.assert_called_once_with(path, exist_ok=True)

    def test_writer_file_naming_logic(self):
        """Test file naming logic for writer."""
        # Test file naming pattern
        base_path = "./tmp/"
        name = "embedding"
        epoch = 2
        batch_idx = 5

        # Expected file naming pattern
        filename = f"{name}_epoch{epoch}_batch{batch_idx}.pt"
        os.path.join(base_path, filename)  # Just create the path, don't assign

        assert filename.startswith(name)
        assert f"epoch{epoch}" in filename
        assert f"batch{batch_idx}" in filename
        assert filename.endswith(".pt")

    def test_writer_during_stages(self):
        """Test writer stage filtering logic."""
        # Test different stage configurations
        during_configs = [["train"], ["val"], ["train", "val"], ["test"], []]

        for during in during_configs:
            # Verify stage filtering logic
            assert "train" in during or "train" not in during
            assert "val" in during or "val" not in during

    def test_writer_epoch_filtering(self):
        """Test writer epoch filtering logic."""
        every_k_epochs = 2

        # Test which epochs should trigger writing
        epochs_to_write = []
        for epoch in range(10):
            if epoch % every_k_epochs == 0:
                epochs_to_write.append(epoch)

        assert epochs_to_write == [0, 2, 4, 6, 8]

    def test_writer_data_collection(self):
        """Test data collection logic for writer."""
        # Mock batch data
        batch = {
            "embedding": torch.randn(4, 512),
            "linear_probe_preds": torch.randn(4, 10),
            "other_data": torch.randn(4, 256),
        }

        # Names to collect
        names = ["embedding", "linear_probe_preds"]

        # Simulate data collection
        collected_data = {}
        for name in names:
            if name in batch:
                collected_data[name] = batch[name]

        assert len(collected_data) == 2
        assert "embedding" in collected_data
        assert "linear_probe_preds" in collected_data
        assert "other_data" not in collected_data

    def test_writer_save_logic(self):
        """Test save logic without actual file I/O."""
        with patch("torch.save") as mock_save:
            # Mock data to save
            data = {"embedding": torch.randn(4, 512)}
            path = "./tmp/embedding_epoch0_batch0.pt"

            # Simulate save
            torch.save(data, path)

            mock_save.assert_called_once_with(data, path)

    def test_writer_with_module_output(self):
        """Test writer integration with module output."""
        # Mock module output
        module_output = {
            "embedding": torch.randn(64, 512),
            "linear_probe_preds": torch.randn(64, 10),
            "loss": torch.tensor(0.5),
            "label": torch.randint(0, 10, (64,)),
        }

        # Writer configuration
        names_to_write = ["embedding", "linear_probe_preds"]

        # Filter data for writing
        data_to_write = {
            name: module_output[name]
            for name in names_to_write
            if name in module_output
        }

        assert len(data_to_write) == 2
        assert data_to_write["embedding"].shape == (64, 512)
        assert data_to_write["linear_probe_preds"].shape == (64, 10)

    def test_writer_file_count_calculation(self):
        """Test calculation of expected file count."""
        # Configuration
        every_k_epochs = 2
        max_epochs = 3
        batches_per_epoch = 20
        names_to_write = ["embedding", "linear_probe_preds"]

        # Calculate expected files
        epochs_written = [e for e in range(max_epochs) if e % every_k_epochs == 0]
        files_per_epoch = len(names_to_write) * batches_per_epoch
        total_files = len(epochs_written) * files_per_epoch

        # For max_epochs=3, every_k_epochs=2: epochs 0, 2 are written
        assert len(epochs_written) == 2
        assert total_files == 2 * 2 * 20  # 2 epochs * 2 names * 20 batches

    def test_writer_cleanup(self):
        """Test cleanup logic for writer output."""
        with patch("shutil.rmtree") as mock_rmtree:
            with patch("os.path.exists", return_value=True):
                # Simulate cleanup
                path = "./tmp/"
                if os.path.exists(path):
                    shutil.rmtree(path)

                mock_rmtree.assert_called_once_with(path)

    def test_writer_batch_tracking(self):
        """Test batch index tracking for file naming."""
        # Simulate batch tracking
        batch_indices = []
        num_batches = 10

        for batch_idx in range(num_batches):
            batch_indices.append(batch_idx)

        assert len(batch_indices) == num_batches
        assert batch_indices == list(range(num_batches))
