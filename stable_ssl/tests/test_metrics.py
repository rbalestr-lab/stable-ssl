def test_formtter():
    import optimalssl
    import torchmetrics

    metrics = {
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    }
    optimalssl.callbacks.utils.format_metrics_as_dict(metrics)

    metrics = [
        torchmetrics.classification.MulticlassAccuracy(10),
        torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    ]
    optimalssl.callbacks.utils.format_metrics_as_dict(metrics)
    optimalssl.callbacks.utils.format_metrics_as_dict(metrics[0])
    optimalssl.callbacks.utils.format_metrics_as_dict(
        {"train": metrics, "val": metrics}
    )
