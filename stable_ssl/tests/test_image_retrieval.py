def test_imgret():
    import lightning as pl
    import torch
    import torchmetrics
    from transformers import AutoModel

    import stable_ssl as ossl
    from stable_ssl.data import transforms

    backbone = AutoModel.from_pretrained("facebook/dino-vits16")

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
        transforms.Resize((224, 224), antialias=True),
        transforms.ToImage(mean=mean, std=std),
    )

    imgret_ds = ossl.data.HFDataset(
        path="randall-lab/revisitop",
        name="roxford5k",
        split="qimlist+imlist",
        trust_remote_code=True,
        transform=val_transform,
    )

    imgret_ds.dataset = imgret_ds.dataset.map(
        lambda example: {"is_query": example["query_id"] >= 0}
    )

    val = torch.utils.data.DataLoader(
        dataset=imgret_ds,
        batch_size=1,
        shuffle=False,
        num_workers=10,
    )

    data = ossl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        with torch.inference_mode():
            x = batch["image"]
            cls_embed = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
            batch["embedding"] = cls_embed
        return batch

    module = ossl.Module(
        backbone=ossl.backbone.EvalOnly(backbone), forward=forward, optim=None
    )

    img_ret = ossl.callbacks.ImageRetrieval(
        module,
        "img_ret",
        input="embedding",
        query_col="is_query",
        retrieval_col=["easy", "hard"],
        features_dim=384,
        metrics={
            "mAP": torchmetrics.RetrievalMAP(),
            "R@1": torchmetrics.RetrievalRecall(top_k=1),
            "R@5": torchmetrics.RetrievalRecall(top_k=5),
            "R@10": torchmetrics.RetrievalRecall(top_k=10),
        },
    )

    trainer = pl.Trainer(
        max_epochs=0,
        num_sanity_val_steps=0,
        callbacks=[img_ret],
        precision="16",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()
    manager.validate()


if __name__ == "__main__":
    test_imgret()
