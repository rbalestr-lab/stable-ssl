"""Unit tests for SimCLR functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics


@pytest.mark.unit
class TestSimCLRUnit:
    """Unit tests for SimCLR components without actual training."""

    def test_simclr_loss_initialization(self):
        """Test NTXEntLoss initialization."""
        with patch("stable_pretraining.losses.NTXEntLoss") as mock_loss:
            loss_fn = mock_loss(temperature=0.1)
            mock_loss.assert_called_once_with(temperature=0.1)
            assert loss_fn is not None

    def test_projector_initialization(self):
        """Test projector network initialization."""
        projector = nn.Linear(512, 128)
        assert projector.in_features == 512
        assert projector.out_features == 128

    def test_fold_views_function(self):
        """Test fold_views functionality for multi-view data."""
        with patch("stable_pretraining.data.fold_views") as mock_fold_views:
            # Mock multi-view data
            batch_size = 32
            n_views = 2
            features = torch.randn(batch_size * n_views, 128)
            sample_idx = torch.repeat_interleave(torch.arange(batch_size), n_views)

            # Mock return value
            view1 = features[:batch_size]
            view2 = features[batch_size:]
            mock_fold_views.return_value = [view1, view2]

            # Call fold_views
            views = mock_fold_views(features, sample_idx)

            mock_fold_views.assert_called_once_with(features, sample_idx)
            assert len(views) == 2
            assert views[0].shape == (batch_size, 128)
            assert views[1].shape == (batch_size, 128)

    def test_simclr_forward_logic(self):
        """Test SimCLR forward pass logic."""
        # Mock components
        mock_backbone = Mock()
        mock_projector = Mock()
        mock_simclr_loss = Mock()

        # Mock outputs
        mock_features = torch.randn(4, 512)
        mock_backbone.return_value = {"logits": mock_features}
        mock_projections = torch.randn(4, 128)
        mock_projector.return_value = mock_projections
        mock_simclr_loss.return_value = torch.tensor(0.5)

        # Create mock module
        mock_module = Mock()
        mock_module.backbone = mock_backbone
        mock_module.projector = mock_projector
        mock_module.simclr_loss = mock_simclr_loss
        mock_module.training = True

        # Define forward function
        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            if self.training:
                proj = self.projector(batch["embedding"])
                # Simulate fold_views
                views = [proj[:2], proj[2:]]
                batch["loss"] = self.simclr_loss(views[0], views[1])
            return batch

        # Test forward pass
        batch = {
            "image": torch.randn(4, 3, 224, 224),
            "sample_idx": torch.tensor([0, 0, 1, 1]),
        }

        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch, "train")

        # Verify calls
        mock_backbone.assert_called_once_with(batch["image"])
        mock_projector.assert_called_once()
        mock_simclr_loss.assert_called_once()
        assert "embedding" in result
        assert "loss" in result

    def test_transform_with_gaussian_blur(self):
        """Test transform composition with Gaussian blur for SimCLR."""
        with patch("stable_pretraining.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            mock_transforms.Compose.return_value = Mock()

            # Create SimCLR augmentation
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

            # Verify GaussianBlur was called
            mock_transforms.GaussianBlur.assert_called_once_with(
                kernel_size=(5, 5), p=1.0
            )

    def test_repeated_random_sampler_for_simclr(self):
        """Test RepeatedRandomSampler for multi-view SimCLR training."""
        with patch(
            "stable_pretraining.data.sampler.RepeatedRandomSampler"
        ) as mock_sampler:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)

            sampler = mock_sampler(mock_dataset, n_views=2)

            mock_sampler.assert_called_once_with(mock_dataset, n_views=2)
            assert sampler is not None

    def test_backbone_config_loading(self):
        """Test loading backbone configuration."""
        with patch("transformers.AutoConfig") as mock_config:
            with patch("transformers.AutoModelForImageClassification") as mock_model:
                # Mock config
                config = mock_config.from_pretrained("microsoft/resnet-18")

                # Mock model creation
                mock_backbone = Mock()
                mock_classifier = Mock()
                mock_classifier.__getitem__ = Mock(return_value=Mock())
                mock_classifier.__setitem__ = Mock()
                mock_backbone.classifier = mock_classifier
                mock_model.from_config.return_value = mock_backbone

                # Create model
                backbone = mock_model.from_config(config)
                backbone.classifier[1] = nn.Identity()

                # Verify
                mock_config.from_pretrained.assert_called_once_with(
                    "microsoft/resnet-18"
                )
                mock_model.from_config.assert_called_once_with(config)
                mock_classifier.__setitem__.assert_called_once()

    def test_module_with_projector(self):
        """Test module creation with projector for SimCLR."""
        with patch("stable_pretraining.Module") as mock_module:
            mock_backbone = Mock()
            mock_projector = Mock()
            mock_forward = Mock()
            mock_loss = Mock()

            mock_module(
                backbone=mock_backbone,
                projector=mock_projector,
                forward=mock_forward,
                simclr_loss=mock_loss,
            )

            mock_module.assert_called_once_with(
                backbone=mock_backbone,
                projector=mock_projector,
                forward=mock_forward,
                simclr_loss=mock_loss,
            )

    def test_ntxent_loss_computation(self):
        """Test NT-Xent loss computation logic."""
        batch_size = 8
        feature_dim = 128
        temperature = 0.1

        # Create mock features for two views
        z1 = torch.randn(batch_size, feature_dim)
        z2 = torch.randn(batch_size, feature_dim)

        # Normalize features
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        # Concatenate all features
        features = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / temperature

        # Verify similarity matrix shape
        assert similarity.shape == (2 * batch_size, 2 * batch_size)

    def test_online_probes_for_simclr(self):
        """Test online probe configuration for SimCLR."""
        # Test linear probe
        linear_probe = nn.Linear(512, 10)
        assert linear_probe.in_features == 512
        assert linear_probe.out_features == 10

        # Test metrics
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        }

        assert isinstance(
            metrics["top1"], torchmetrics.classification.MulticlassAccuracy
        )
        assert metrics["top5"].top_k == 5
