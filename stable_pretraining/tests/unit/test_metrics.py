"""Unit tests for metrics formatting utilities."""

import pytest
import torchmetrics

import stable_pretraining


@pytest.mark.unit
class TestMetricsFormatter:
    """Test the format_metrics_as_dict utility function."""

    def test_format_dict_metrics(self):
        """Test formatting metrics provided as a dictionary."""
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        }
        result = stable_pretraining.callbacks.utils.format_metrics_as_dict(metrics)
        # The function returns a ModuleDict, not a regular dict
        from torch.nn import ModuleDict

        assert isinstance(result, ModuleDict)
        # Check for expected structure
        assert hasattr(result, "_val") or hasattr(result, "_train")

    def test_format_list_metrics(self):
        """Test formatting metrics provided as a list."""
        metrics = [
            torchmetrics.classification.MulticlassAccuracy(10),
            torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        ]
        result = stable_pretraining.callbacks.utils.format_metrics_as_dict(metrics)
        from torch.nn import ModuleDict

        assert isinstance(result, ModuleDict)

    def test_format_single_metric(self):
        """Test formatting a single metric."""
        metric = torchmetrics.classification.MulticlassAccuracy(10)
        result = stable_pretraining.callbacks.utils.format_metrics_as_dict(metric)
        from torch.nn import ModuleDict

        assert isinstance(result, ModuleDict)

    def test_format_nested_metrics(self):
        """Test formatting nested metrics dictionaries."""
        metrics = [
            torchmetrics.classification.MulticlassAccuracy(10),
            torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        ]
        nested = {"train": metrics, "val": metrics}
        result = stable_pretraining.callbacks.utils.format_metrics_as_dict(nested)
        from torch.nn import ModuleDict

        assert isinstance(result, ModuleDict)
        # Should have _train and _val attributes
        assert hasattr(result, "_train")
        assert hasattr(result, "_val")
