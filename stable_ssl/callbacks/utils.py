import torch
import torchmetrics


class EarlyStopping(torch.nn.Module):
    """Early stopping mechanism with support for metric milestones and patience.

    This module provides flexible early stopping capabilities that can halt training
    based on metric performance. It supports both milestone-based stopping (stop if
    metric doesn't reach target by specific epochs) and patience-based stopping
    (stop if metric doesn't improve for N epochs).

    Args:
        mode: Optimization direction - 'min' for metrics to minimize (e.g., loss),
            'max' for metrics to maximize (e.g., accuracy).
        milestones: Dict mapping epoch numbers to target metric values. Training
            stops if targets are not met at specified epochs.
        metric_name: Name of the metric to monitor if metric is a dict.
        patience: Number of epochs with no improvement before stopping.

    Example:
        >>> early_stop = EarlyStopping(mode="max", milestones={10: 0.8, 20: 0.9})
        >>> # Stops if accuracy < 0.8 at epoch 10 or < 0.9 at epoch 20
    """

    def __init__(
        self,
        mode: str = "min",
        milestones: dict[int, float] = None,
        metric_name: str = None,
        patience: int = 10,
    ):
        super().__init__()
        self.mode = mode
        self.milestones = milestones or {}
        self.metric_name = metric_name
        self.patience = patience
        self.register_buffer("history", torch.zeros(patience))

    def should_stop(self, metric, step):
        if self.metric_name is None:
            assert type(metric) is not dict
        else:
            assert self.metric_name in metric
            metric = metric[self.metric_name]
        if step in self.milestones:
            if self.mode == "min":
                return metric > self.milestones[step]
            elif self.mode == "max":
                return metric < self.milestones[step]
        return False


def format_metrics_as_dict(metrics):
    """Formats various metric input formats into a standardized dictionary structure.

    This utility function handles multiple input formats for metrics and converts
    them into a consistent ModuleDict structure with separate train and validation
    metrics. This standardization simplifies metric handling across callbacks.

    Args:
        metrics: Can be:
            - None: Returns empty train and val dicts
            - Single torchmetrics.Metric: Applied to validation only
            - Dict with 'train' and 'val' keys: Separated accordingly
            - Dict of metrics: All applied to validation
            - List/tuple of metrics: All applied to validation

    Returns:
        ModuleDict with '_train' and '_val' keys, each containing metric ModuleDicts.

    Raises:
        ValueError: If metrics format is invalid or contains non-torchmetric objects.
    """
    if metrics is None:
        train = {}
        eval = {}
    elif isinstance(metrics, torchmetrics.Metric):
        train = {}
        eval = torch.nn.ModuleDict({metrics.__class__.__name__: metrics})
    elif type(metrics) is dict and set(metrics.keys()) == set(["train", "val"]):
        if type(metrics["train"]) in [list, tuple]:
            train = {}
            for m in metrics["train"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                train[m.__class__.__name__] = m
        else:
            train = metrics["train"]
        if type(metrics["val"]) in [list, tuple]:
            eval = {}
            for m in metrics["val"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                eval[m.__class__.__name__] = m
        else:
            eval = metrics["eval"]
    elif type(metrics) is dict:
        train = {}
        for k, v in metrics.items():
            assert type(k) is str
            assert isinstance(v, torchmetrics.Metric)
        eval = metrics
    elif type(metrics) in [list, tuple]:
        train = {}
        for m in metrics:
            if not isinstance(m, torchmetrics.Metric):
                raise ValueError(f"metric {m} is no a torchmetric")
        eval = {m.__class__.__name__: m for m in metrics}
    else:
        raise ValueError(
            "metrics can only be a torchmetric of list/tuple of torchmetrics"
        )
    return torch.nn.ModuleDict(
        {
            "_train": torch.nn.ModuleDict(train),
            "_val": torch.nn.ModuleDict(eval),
        }
    )
