def test_dictconfig():
    import logging

    import lightning as pl
    import stable_ssl as ossl
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)

    # without transform
    data = OmegaConf.create(
        {
            "_target_": "stable_ssl.data.DataModule",
            "train": {
                "dataset": {
                    "_target_": "stable_ssl.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "test",
                    "transform": {
                        "_target_": "stable_ssl.data.transforms.ToImage",
                    },
                },
                "batch_size": 20,
                "num_workers": 10,
                "drop_last": True,
                "shuffle": True,
            },
            "test": {
                "dataset": {
                    "_target_": "stable_ssl.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "test",
                    "transform": {
                        "_target_": "stable_ssl.data.transforms.ToImage",
                    },
                },
                "batch_size": 20,
                "num_workers": 10,
            },
        }
    )
    manager = ossl.Manager(
        trainer=pl.Trainer(max_epochs=1), module=pl.LightningModule(), data=data
    )


def test_datamodule():
    import logging

    import lightning as pl
    import stable_ssl as ossl

    logging.basicConfig(level=logging.INFO)

    # without transform
    train = dict(
        dataset=ossl.data.HFDataset(
            path="ylecun/mnist", split="train", transform=ossl.data.transforms.ToImage()
        ),
        batch_size=512,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    test = dict(
        dataset=ossl.data.HFDataset(
            path="ylecun/mnist", split="test", transform=ossl.data.transforms.ToImage()
        ),
        batch_size=20,
        num_workers=10,
    )
    data = ossl.data.DataModule(train=train, test=test)
    manager = ossl.Manager(trainer=pl.Trainer(), module=pl.LightningModule(), data=data)


def test_dataloader():
    import logging

    import lightning as pl
    import stable_ssl as ossl
    import torch

    logging.basicConfig(level=logging.INFO)

    # without transform
    train = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="ylecun/mnist", split="train", transform=ossl.data.transforms.ToImage()
        ),
        batch_size=512,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    val = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="ylecun/mnist", split="test", transform=ossl.data.transforms.ToImage()
        ),
        batch_size=20,
        num_workers=10,
    )
    data = ossl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        x = batch["image"]
        preds = self.backbone(x)
        batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
        self.log(value=batch["loss"], name="loss", on_step=True, on_epoch=False)
        acc = preds.argmax(1).eq(batch["label"]).float().mean() * 100
        self.log(value=acc, name="acc", on_step=True, on_epoch=True)
        return batch

    backbone = ossl.backbone.Resnet9(num_classes=10, num_channels=1)

    module = ossl.Module(backbone=backbone, forward=forward)

    manager = ossl.Manager(
        trainer=pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=1,
            logger=False,
            enable_checkpointing=False,
        ),
        module=module,
        data=data,
    )
    manager()


if __name__ == "__main__":
    test_dictconfig()
    test_datamodule()
    test_dataloader()
