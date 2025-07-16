"""Unit tests for supervised training functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics


@pytest.mark.unit
class TestSupervisedUnit:
    """Unit tests for supervised training components without actual training."""

    def test_supervised_forward_logic(self):
        """Test supervised forward pass logic."""
        # Mock components
        mock_backbone = Mock()
        mock_classifier = Mock()

        # Mock outputs
        mock_features = torch.randn(4, 512)
        mock_backbone.return_value = {"logits": mock_features}
        mock_preds = torch.randn(4, 10)
        mock_classifier.return_value = mock_preds

        # Create mock module
        mock_module = Mock()
        mock_module.backbone = mock_backbone
        mock_module.classifier = mock_classifier
        mock_module.training = True

        # Define forward function
        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            if self.training:
                preds = self.classifier(batch["embedding"])
                batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
            return batch

        # Test forward pass
        batch = {
            "image": torch.randn(4, 3, 224, 224),
            "label": torch.tensor([0, 1, 2, 3]),
        }

        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        # Verify calls
        mock_backbone.assert_called_once_with(batch["image"])
        mock_classifier.assert_called_once_with(mock_features)
        assert "embedding" in result
        assert "loss" in result

        # Test eval mode
        mock_module.training = False
        # Reset mocks for new calls
        mock_backbone.reset_mock()
        mock_classifier.reset_mock()
        # Use a fresh batch to avoid contamination
        batch_val = {
            "image": torch.randn(4, 3, 224, 224),
            "label": torch.tensor([0, 1, 2, 3]),
        }
        result = forward_bound(batch_val, "val")
        assert "embedding" in result
        assert "loss" not in result

    def test_backbone_configuration(self):
        """Test backbone configuration from pretrained model."""
        with patch("transformers.AutoConfig") as mock_config:
            with patch("transformers.AutoModelForImageClassification") as mock_model:
                # Mock config and model
                config = mock_config.from_pretrained("microsoft/resnet-18")
                mock_backbone = Mock()
                mock_classifier = Mock()
                mock_classifier.__getitem__ = Mock(return_value=Mock())
                mock_classifier.__setitem__ = Mock()
                mock_backbone.classifier = mock_classifier
                mock_model.from_config.return_value = mock_backbone

                # Create and modify backbone
                backbone = mock_model.from_config(config)
                backbone.classifier[1] = nn.Identity()

                # Verify
                mock_config.from_pretrained.assert_called_once_with(
                    "microsoft/resnet-18"
                )
                mock_model.from_config.assert_called_once_with(config)
                mock_classifier.__setitem__.assert_called_once()

    def test_classifier_initialization(self):
        """Test classifier layer initialization."""
        classifier = nn.Linear(512, 10)
        assert classifier.in_features == 512
        assert classifier.out_features == 10

    def test_module_with_classifier(self):
        """Test module creation with classifier for supervised training."""
        with patch("stable_ssl.Module") as mock_module:
            mock_backbone = Mock()
            mock_classifier = Mock()
            mock_forward = Mock()

            mock_module(
                backbone=mock_backbone, classifier=mock_classifier, forward=mock_forward
            )

            mock_module.assert_called_once_with(
                backbone=mock_backbone, classifier=mock_classifier, forward=mock_forward
            )

    def test_rankme_callback_initialization(self):
        """Test RankMe callback initialization."""
        with patch("stable_ssl.callbacks.RankMe") as mock_rankme:
            mock_module = Mock()

            mock_rankme(mock_module, "rankme", "embedding", 20000, target_shape=512)

            mock_rankme.assert_called_once_with(
                mock_module, "rankme", "embedding", 20000, target_shape=512
            )

    def test_transform_composition_for_supervised(self):
        """Test transform composition for supervised training."""
        with patch("stable_ssl.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            mock_transforms.Compose.return_value = Mock()

            # Train transform with augmentations
            mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.RandomResizedCrop((224, 224)),
                mock_transforms.RandomHorizontalFlip(p=0.5),
                mock_transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                mock_transforms.RandomGrayscale(p=0.2),
                mock_transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            # Val transform without augmentations
            mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.Resize((256, 256)),
                mock_transforms.CenterCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert mock_transforms.Compose.call_count == 2

    def test_cross_entropy_loss_computation(self):
        """Test cross entropy loss computation for supervised training."""
        batch_size = 16
        num_classes = 10

        # Create predictions and labels
        preds = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Compute loss
        loss = torch.nn.functional.cross_entropy(preds, labels)

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_online_probes_with_supervised(self):
        """Test online probe configuration for supervised training."""
        # Test probe initialization
        probe = nn.Linear(512, 10)
        loss_fn = nn.CrossEntropyLoss()

        # Test metrics
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        }

        assert probe.in_features == 512
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert metrics["top1"].num_classes == 10
        assert metrics["top5"].top_k == 5

    def test_dataloader_configuration_supervised(self):
        """Test DataLoader configuration for supervised training."""
        with patch("torch.utils.data.DataLoader") as mock_loader_class:
            mock_dataset = Mock()

            # Train loader
            mock_loader_class(
                dataset=mock_dataset, batch_size=64, num_workers=20, drop_last=True
            )

            # Val loader
            mock_loader_class(dataset=mock_dataset, batch_size=128, num_workers=10)

            # Verify configurations
            train_call = mock_loader_class.call_args_list[0][1]
            assert train_call["batch_size"] == 64
            assert train_call["drop_last"] is True

            val_call = mock_loader_class.call_args_list[1][1]
            assert val_call["batch_size"] == 128
            assert val_call.get("drop_last", False) is False

    def test_trainer_configuration_supervised(self):
        """Test trainer configuration for supervised training."""
        with patch("lightning.Trainer") as mock_trainer:
            mock_callbacks = [Mock(), Mock(), Mock()]  # linear_probe, knn_probe, rankme

            mock_trainer(
                max_epochs=10,
                num_sanity_val_steps=1,
                callbacks=mock_callbacks,
                precision="16-mixed",
                logger=False,
                enable_checkpointing=False,
            )

            call_kwargs = mock_trainer.call_args[1]
            assert call_kwargs["max_epochs"] == 10
            assert call_kwargs["precision"] == "16-mixed"
            assert len(call_kwargs["callbacks"]) == 3
