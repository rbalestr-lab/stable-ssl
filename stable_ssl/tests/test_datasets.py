def test_hf_datasets():
    import torch
    from torchvision.transforms import v2

    import stable_ssl as ossl

    # without transform
    dataset1 = ossl.data.HFDataset("ylecun/mnist", split="train")
    # with transform
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def t(x):
        x["image"] = transform(x["image"])
        return x

    dataset2 = ossl.data.HFDataset("ylecun/mnist", split="train", transform=t)

    assert transform(dataset1[0]["image"]).eq(dataset2[0]["image"]).all()

    dataset3 = ossl.data.HFDataset(
        "ylecun/mnist", split="train", rename_columns=dict(image="toto")
    )
    assert transform(dataset3[0]["toto"]).eq(dataset2[0]["image"]).all()


def test_hf_dataloaders():
    import torch
    from torchvision.transforms import v2

    import stable_ssl as ossl

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def t(x):
        x["image"] = transform(x["image"])
        return x

    # without transform
    dataset = ossl.data.HFDataset("ylecun/mnist", split="train", transform=t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    for x in loader:
        assert x["image"].shape == (4, 1, 28, 28)
        assert len(x["label"]) == 4
        break


def test_datamodule():
    import logging

    import torch
    from omegaconf import OmegaConf

    import stable_ssl as ossl

    logging.basicConfig(level=logging.INFO)

    # without transform
    train = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_ssl.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "train",
            },
            "batch_size": 20,
            "drop_last": True,
        }
    )
    test = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_ssl.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "test",
                "transform": {
                    "_target_": "stable_ssl.data.transforms.ToImage",
                },
            },
            "batch_size": 20,
        }
    )
    module = ossl.data.DataModule(train=train, test=test, val=test, predict=test)
    module.prepare_data()
    module.setup("fit")
    assert not torch.is_tensor(module.train_dataset[0]["image"])
    loader = module.train_dataloader()
    assert loader.drop_last
    module.setup("test")
    loader = module.test_dataloader()
    assert torch.is_tensor(module.test_dataset[0]["image"])
    assert not loader.drop_last
    module.setup("validate")
    loader = module.val_dataloader()
    assert not loader.drop_last
    module.setup("predict")
    loader = module.predict_dataloader()
    assert not loader.drop_last


# if __name__ == "__main__":
# test_datamodule()
# test_hf_dataloaders()
