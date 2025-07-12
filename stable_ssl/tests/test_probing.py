def test_probing():
    import lightning as pl
    import torch
    import torchmetrics
    from transformers import AutoModelForImageClassification

    import stable_ssl as ossl
    from stable_ssl.data import transforms

    # without transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToImage(mean=mean, std=std),
    )
    train = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train[:128]",
            transform=train_transform,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(mean=mean, std=std),
    )
    val = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="validation[:128]",
            transform=val_transform,
        ),
        batch_size=128,
        num_workers=10,
    )

    data = ossl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        with torch.inference_mode():
            x = batch["image"]
            batch["embedding"] = self.backbone(x)["logits"]
        return batch

    backbone = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    backbone.classifier[1] = torch.nn.Identity()
    module = ossl.Module(
        backbone=ossl.backbone.EvalOnly(backbone), forward=forward, optim=None
    )
    linear_probe = ossl.callbacks.OnlineProbe(
        "linear_probe",
        module,
        "embedding",
        "label",
        probe=torch.nn.Linear(512, 10),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=torchmetrics.classification.MulticlassAccuracy(10),
    )
    knn_probe = ossl.callbacks.OnlineKNN(
        module,
        "knn_probe",
        "embedding",
        "label",
        50000,
        metrics=torchmetrics.classification.MulticlassAccuracy(10),
        k=10,
        features_dim=512,
    )

    trainer = pl.Trainer(
        max_steps=10,
        num_sanity_val_steps=1,
        callbacks=[linear_probe, knn_probe],
        precision="16",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()
    manager.validate()


if __name__ == "__main__":
    test_probing()
