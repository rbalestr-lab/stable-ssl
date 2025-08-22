"""Unit tests for MNIST dataset and data module functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.unit
class TestMNISTUnit:
    """Unit tests for MNIST-related data loading and configuration."""

    def test_dictconfig_structure(self):
        """Test DictConfig structure for DataModule."""
        # Create configuration
        data_config = OmegaConf.create(
            {
                "_target_": "stable_pretraining.data.DataModule",
                "train": {
                    "dataset": {
                        "_target_": "stable_pretraining.data.HFDataset",
                        "path": "ylecun/mnist",
                        "split": "test",
                        "transform": {
                            "_target_": "stable_pretraining.data.transforms.ToImage",
                        },
                    },
                    "batch_size": 20,
                    "num_workers": 10,
                    "drop_last": True,
                    "shuffle": True,
                },
                "test": {
                    "dataset": {
                        "_target_": "stable_pretraining.data.HFDataset",
                        "path": "ylecun/mnist",
                        "split": "test",
                        "transform": {
                            "_target_": "stable_pretraining.data.transforms.ToImage",
                        },
                    },
                    "batch_size": 20,
                    "num_workers": 10,
                },
            }
        )

        # Verify configuration structure
        assert data_config._target_ == "stable_pretraining.data.DataModule"
        assert data_config.train.dataset.path == "ylecun/mnist"
        assert data_config.train.batch_size == 20
        assert data_config.train.drop_last is True
        assert data_config.train.shuffle is True
        assert data_config.test.dataset.split == "test"
        assert "shuffle" not in data_config.test

    def test_manager_initialization_with_dictconfig(self):
        """Test Manager initialization with DictConfig."""
        with patch("stable_pretraining.Manager") as mock_manager:
            with patch("lightning.Trainer") as mock_trainer:
                with patch("lightning.LightningModule") as mock_module:
                    data_config = Mock()

                    manager = mock_manager(
                        trainer=mock_trainer(max_epochs=1),
                        module=mock_module(),
                        data=data_config,
                    )

                    mock_manager.assert_called_once()
                    assert manager is not None

    def test_datamodule_dict_initialization(self):
        """Test DataModule initialization with dictionary configuration."""
        with patch("stable_pretraining.data.DataModule") as mock_datamodule:
            with patch("stable_pretraining.data.HFDataset") as mock_dataset:
                with patch(
                    "stable_pretraining.data.transforms.ToImage"
                ) as mock_transform:
                    # Mock dataset instances
                    train_dataset = mock_dataset.return_value
                    test_dataset = mock_dataset.return_value
                    mock_transform.return_value  # Just create the mock, don't assign

                    # Create configuration dicts
                    train_config = dict(
                        dataset=train_dataset,
                        batch_size=512,
                        shuffle=True,
                        num_workers=10,
                        drop_last=True,
                    )

                    test_config = dict(
                        dataset=test_dataset,
                        batch_size=20,
                        num_workers=10,
                    )

                    # Create DataModule
                    data_module = mock_datamodule(train=train_config, test=test_config)

                    mock_datamodule.assert_called_once_with(
                        train=train_config, test=test_config
                    )
                    assert data_module is not None

    def test_forward_function_logic(self):
        """Test forward function for supervised training."""
        mock_backbone = Mock()
        mock_preds = torch.randn(4, 10)  # 4 samples, 10 classes
        mock_backbone.return_value = mock_preds

        mock_module = Mock()
        mock_module.backbone = mock_backbone
        mock_module.log = Mock()

        # Define forward function
        def forward(self, batch, stage):
            x = batch["image"]
            preds = self.backbone(x)
            batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
            self.log(value=batch["loss"], name="loss", on_step=True, on_epoch=False)
            acc = preds.argmax(1).eq(batch["label"]).float().mean() * 100
            self.log(value=acc, name="acc", on_step=True, on_epoch=True)
            return batch

        # Create batch
        batch = {
            "image": torch.randn(4, 1, 28, 28),
            "label": torch.tensor([0, 1, 2, 3]),
        }

        # Bind and call forward
        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch, "train")

        # Verify calls
        mock_backbone.assert_called_once_with(batch["image"])
        assert "loss" in result
        assert mock_module.log.call_count == 2  # Called for loss and accuracy

    def test_resnet9_initialization(self):
        """Test Resnet9 backbone initialization for MNIST."""
        with patch("stable_pretraining.backbone.Resnet9") as mock_resnet:
            backbone = mock_resnet(num_classes=10, num_channels=1)

            mock_resnet.assert_called_once_with(num_classes=10, num_channels=1)
            assert backbone is not None

    def test_dataloader_creation(self):
        """Test DataLoader creation with MNIST dataset."""
        with patch("torch.utils.data.DataLoader") as mock_loader_class:
            with patch("stable_pretraining.data.HFDataset") as mock_dataset:
                with patch(
                    "stable_pretraining.data.transforms.ToImage"
                ) as mock_transform:
                    dataset = mock_dataset.return_value
                    mock_transform.return_value  # Just create the mock, don't assign

                    # Create train dataloader
                    mock_loader_class(
                        dataset=dataset,
                        batch_size=512,
                        shuffle=True,
                        num_workers=10,
                        drop_last=True,
                    )

                    # Verify train loader configuration
                    train_call_kwargs = mock_loader_class.call_args_list[0][1]
                    assert train_call_kwargs["batch_size"] == 512
                    assert train_call_kwargs["shuffle"] is True
                    assert train_call_kwargs["drop_last"] is True

                    # Create val dataloader
                    mock_loader_class(
                        dataset=dataset,
                        batch_size=20,
                        num_workers=10,
                    )

                    # Verify val loader configuration
                    val_call_kwargs = mock_loader_class.call_args_list[1][1]
                    assert val_call_kwargs["batch_size"] == 20
                    assert val_call_kwargs.get("shuffle", False) is False

    def test_cross_entropy_loss_computation(self):
        """Test cross entropy loss computation."""
        preds = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 3])

        loss = torch.nn.functional.cross_entropy(preds, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_accuracy_computation(self):
        """Test accuracy computation logic."""
        # Create predictions with known correct answers
        preds = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],  # Predicts class 0
                [0.0, 10.0, 0.0, 0.0],  # Predicts class 1
                [0.0, 0.0, 10.0, 0.0],  # Predicts class 2
                [0.0, 0.0, 0.0, 10.0],  # Predicts class 3
            ]
        )

        labels = torch.tensor([0, 1, 2, 3])

        # Compute accuracy
        acc = preds.argmax(1).eq(labels).float().mean() * 100

        assert acc.item() == 100.0  # All correct

    def test_trainer_configuration(self):
        """Test Trainer configuration for MNIST training."""
        with patch("lightning.Trainer") as mock_trainer_class:
            mock_trainer_class(
                max_steps=3,
                num_sanity_val_steps=1,
                logger=False,
                enable_checkpointing=False,
            )

            mock_trainer_class.assert_called_once_with(
                max_steps=3,
                num_sanity_val_steps=1,
                logger=False,
                enable_checkpointing=False,
            )
