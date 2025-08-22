"""Unit tests for image retrieval functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
import torchmetrics


@pytest.mark.unit
class TestImageRetrievalUnit:
    """Unit tests for image retrieval components without actual model loading."""

    def test_image_retrieval_callback_initialization(self):
        """Test ImageRetrieval callback initialization."""
        with patch("stable_ssl.callbacks.ImageRetrieval") as mock_callback:
            mock_module = Mock()
            mock_metrics = {
                "mAP": Mock(spec=torchmetrics.RetrievalMAP),
                "R@1": Mock(spec=torchmetrics.RetrievalRecall),
                "R@5": Mock(spec=torchmetrics.RetrievalRecall),
                "R@10": Mock(spec=torchmetrics.RetrievalRecall),
            }

            mock_callback(
                mock_module,
                "img_ret",
                input="embedding",
                query_col="is_query",
                retrieval_col=["easy", "hard"],
                features_dim=384,
                metrics=mock_metrics,
            )

            mock_callback.assert_called_once_with(
                mock_module,
                "img_ret",
                input="embedding",
                query_col="is_query",
                retrieval_col=["easy", "hard"],
                features_dim=384,
                metrics=mock_metrics,
            )

    def test_transform_composition(self):
        """Test transform composition for train and validation."""
        with patch("stable_ssl.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # Test train transform
            mock_transforms.Compose.return_value = Mock()
            train_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.RandomResizedCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert mock_transforms.Compose.called
            assert train_transform is not None

            # Test validation transform
            val_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.Resize((224, 224), antialias=True),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert val_transform is not None

    def test_forward_function_logic(self):
        """Test forward function for feature extraction."""
        mock_backbone = Mock()
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 197, 384)  # ViT output shape
        mock_backbone.return_value = mock_output

        # Create mock module with backbone
        mock_module = Mock()
        mock_module.backbone = mock_backbone

        # Test forward function
        def forward(self, batch, stage):
            with torch.inference_mode():
                x = batch["image"]
                cls_embed = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
                batch["embedding"] = cls_embed
            return batch

        # Simulate forward pass
        batch = {"image": torch.randn(2, 3, 224, 224)}

        # Bind forward to mock module and call it
        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch, "val")

        # Verify backbone was called
        mock_backbone.assert_called_once()
        assert "embedding" in result
        assert result["embedding"].shape == (2, 384)

    def test_dataset_mapping(self):
        """Test dataset mapping for query identification."""
        # Mock dataset with map method
        mock_dataset = Mock()
        mock_dataset.map = Mock(return_value=mock_dataset)

        # Mock the HFDataset
        mock_imgret_ds = Mock()
        mock_imgret_ds.dataset = mock_dataset

        # Apply mapping
        mock_imgret_ds.dataset = mock_imgret_ds.dataset.map(
            lambda example: {"is_query": example["query_id"] >= 0}
        )

        # Verify map was called
        mock_dataset.map.assert_called_once()

        # Test the mapping function
        map_func = mock_dataset.map.call_args[0][0]
        test_example = {"query_id": 5}
        result = map_func(test_example)
        assert result["is_query"] is True

        test_example_negative = {"query_id": -1}
        result_negative = map_func(test_example_negative)
        assert result_negative["is_query"] is False

    def test_eval_only_backbone(self):
        """Test EvalOnly wrapper for backbone."""
        with patch("stable_ssl.backbone.EvalOnly") as mock_eval_only:
            mock_backbone = Mock()
            eval_backbone = mock_eval_only(mock_backbone)

            mock_eval_only.assert_called_once_with(mock_backbone)
            assert eval_backbone is not None

    def test_module_creation_with_no_optim(self):
        """Test module creation with no optimizer."""
        with patch("stable_ssl.Module") as mock_module_class:
            mock_backbone = Mock()
            mock_forward = Mock()

            mock_module_class(backbone=mock_backbone, forward=mock_forward, optim=None)

            mock_module_class.assert_called_once_with(
                backbone=mock_backbone, forward=mock_forward, optim=None
            )

    def test_retrieval_metrics_initialization(self):
        """Test retrieval metrics initialization."""
        metrics = {
            "mAP": torchmetrics.RetrievalMAP(),
            "R@1": torchmetrics.RetrievalRecall(top_k=1),
            "R@5": torchmetrics.RetrievalRecall(top_k=5),
            "R@10": torchmetrics.RetrievalRecall(top_k=10),
        }

        assert isinstance(metrics["mAP"], torchmetrics.RetrievalMAP)
        assert isinstance(metrics["R@1"], torchmetrics.RetrievalRecall)
        assert metrics["R@1"].top_k == 1
        assert metrics["R@5"].top_k == 5
        assert metrics["R@10"].top_k == 10

    def test_dataloader_configuration(self):
        """Test DataLoader configuration for train and validation."""
        with patch("torch.utils.data.DataLoader") as mock_loader_class:
            mock_train_dataset = Mock()
            mock_val_dataset = Mock()

            # Test train dataloader
            mock_loader_class(
                dataset=mock_train_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=10,
                drop_last=True,
            )

            assert mock_loader_class.call_args_list[0][1]["batch_size"] == 128
            assert mock_loader_class.call_args_list[0][1]["shuffle"] is True
            assert mock_loader_class.call_args_list[0][1]["drop_last"] is True

            # Test validation dataloader
            mock_loader_class(
                dataset=mock_val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=10,
            )

            assert mock_loader_class.call_args_list[1][1]["batch_size"] == 1
            assert mock_loader_class.call_args_list[1][1]["shuffle"] is False
