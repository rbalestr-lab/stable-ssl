import torch
import torchmetrics


class EarlyStopping(torch.nn.Module):
    def __init__(
        self,
        mode: str = "min",
        milestones: dict[int, float] = None,
        metric_name: str = None,
    ):
        self.mode = mode
        self.milestones = milestones or {}
        self.metric_name = metric_name
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
    print(train, eval)
    return torch.nn.ModuleDict(
        {
            "_train": torch.nn.ModuleDict(train),
            "_val": torch.nn.ModuleDict(eval),
        }
    )
