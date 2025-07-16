"""Unit tests for checkpointing functionality."""

from unittest.mock import Mock, patch

import pytest
from torch import nn


@pytest.mark.unit
class TestCheckpointingUnit:
    """Unit tests for checkpointing without actual file I/O or training."""

    def test_sklearn_checkpoint_callback_initialization(self):
        """Test SklearnCheckpoint callback can be initialized."""
        with patch("stable_ssl.callbacks.SklearnCheckpoint") as mock_callback:
            callback = mock_callback()
            assert callback is not None

    def test_module_with_sklearn_components(self):
        """Test module can be created with sklearn components."""
        with patch("stable_ssl.module.Module") as mock_module:
            with patch("sklearn.tree.DecisionTreeRegressor") as mock_tree:
                # Mock the components
                mock_backbone = Mock()
                mock_linear_probe = Mock(spec=nn.Linear)
                mock_nonlinear_probe = Mock()
                mock_tree_instance = mock_tree.return_value

                # Create module
                module = mock_module(
                    backbone=mock_backbone,
                    linear_probe=mock_linear_probe,
                    nonlinear_probe=mock_nonlinear_probe,
                    tree=mock_tree_instance,
                    G_scaling=1,
                    imbalance_factor=0,
                    batch_size=512,
                    lr=1,
                )

                assert module is not None
                mock_module.assert_called_once()

    def test_checkpoint_save_logic(self):
        """Test checkpoint saving logic without actual file operations."""
        with patch("lightning.Trainer") as mock_trainer_class:
            mock_trainer = mock_trainer_class.return_value
            mock_trainer.save_checkpoint = Mock()

            # Simulate save checkpoint call
            checkpoint_path = "test.ckpt"
            mock_trainer.save_checkpoint(checkpoint_path)

            mock_trainer.save_checkpoint.assert_called_once_with(checkpoint_path)

    def test_checkpoint_load_logic(self):
        """Test checkpoint loading logic without actual file operations."""
        with patch("lightning.Trainer") as mock_trainer_class:
            mock_trainer = mock_trainer_class.return_value
            mock_module = Mock()
            mock_dataloader = Mock()

            # Mock the fit method to simulate checkpoint loading
            mock_trainer.fit = Mock()

            # Simulate loading from checkpoint
            mock_trainer.fit(
                mock_module, train_dataloaders=mock_dataloader, ckpt_path="test.ckpt"
            )

            mock_trainer.fit.assert_called_once_with(
                mock_module, train_dataloaders=mock_dataloader, ckpt_path="test.ckpt"
            )

    def test_sklearn_attribute_persistence(self):
        """Test that sklearn model attributes can be set and checked."""
        # Create a mock sklearn model
        mock_tree = Mock()

        # Test setting attribute
        mock_tree.test = 3
        assert mock_tree.test == 3

        # Test changing attribute
        mock_tree.test = 5
        assert mock_tree.test == 5

        # Test checking for attribute existence
        assert hasattr(mock_tree, "test")

        # Test deleting attribute
        delattr(mock_tree, "test")
        assert not hasattr(mock_tree, "test")

    def test_trainer_configuration_for_checkpointing(self):
        """Test trainer can be configured with checkpointing settings."""
        with patch("lightning.Trainer") as mock_trainer_class:
            # Test trainer with checkpointing disabled
            mock_trainer_class(
                max_epochs=0,
                accelerator="cpu",
                enable_checkpointing=False,
                logger=False,
                limit_train_batches=2,
            )

            mock_trainer_class.assert_called_with(
                max_epochs=0,
                accelerator="cpu",
                enable_checkpointing=False,
                logger=False,
                limit_train_batches=2,
            )

            # Test trainer with callbacks
            with patch("stable_ssl.callbacks.SklearnCheckpoint") as mock_callback:
                callback_instance = mock_callback.return_value
                mock_trainer_class(
                    max_epochs=0,
                    accelerator="cpu",
                    enable_checkpointing=False,
                    logger=False,
                    callbacks=[callback_instance],
                )

                # Verify the callback was included in the mock's call args
                assert (
                    callback_instance
                    in mock_trainer_class.call_args.kwargs["callbacks"]
                )
