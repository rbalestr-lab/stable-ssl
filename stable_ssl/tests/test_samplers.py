import pytest


@pytest.mark.parametrize("n_views", [1, 2, 4])
def test_repeated_sampler(n_views):
    import logging

    import stable_ssl as ossl
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)

    # without partial
    train = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_ssl.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "train[:128]",
                "transform": {
                    "_target_": "stable_ssl.data.transforms.ToImage",
                },
            },
            "sampler": {
                "_target_": "stable_ssl.data.sampler.RepeatedRandomSampler",
                "n_views": n_views,
                "data_source_or_len": 128,
            },
            "batch_size": 128,
        }
    )
    test = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_ssl.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "test[:128]",
                "transform": {
                    "_target_": "stable_ssl.data.transforms.ToImage",
                },
            },
            "sampler": {
                "_target_": "stable_ssl.data.sampler.RepeatedRandomSampler",
                "n_views": n_views,
                "data_source_or_len": 128,
            },
            "batch_size": 128,
        }
    )
    module = ossl.data.DataModule(train=train, test=test, val=test, predict=test)
    module.prepare_data()
    module.setup("fit")
    loader = module.train_dataloader()
    for batch in loader:
        break
    for i in range(len(batch["image"])):
        assert batch["image"][(i // n_views) * n_views].eq(batch["image"][i]).all()
        assert (
            batch["sample_idx"][(i // n_views) * n_views]
            .eq(batch["sample_idx"][i])
            .all()
        )
        assert batch["label"][(i // n_views) * n_views].eq(batch["label"][i]).all()
    module.setup("validate")
    loader = module.val_dataloader()
    for batch in loader:
        break
    for i in range(len(batch["image"])):
        assert batch["image"][(i // n_views) * n_views].eq(batch["image"][i]).all()
        assert (
            batch["sample_idx"][(i // n_views) * n_views]
            .eq(batch["sample_idx"][i])
            .all()
        )
        assert batch["label"][(i // n_views) * n_views].eq(batch["label"][i]).all()
    module.setup("test")
    loader = module.test_dataloader()
    for batch in loader:
        break
    for i in range(len(batch["image"])):
        assert batch["image"][(i // n_views) * n_views].eq(batch["image"][i]).all()
        assert (
            batch["sample_idx"][(i // n_views) * n_views]
            .eq(batch["sample_idx"][i])
            .all()
        )
        assert batch["label"][(i // n_views) * n_views].eq(batch["label"][i]).all()
    module.setup("predict")
    loader = module.predict_dataloader()
    for batch in loader:
        break
    for i in range(len(batch["image"])):
        assert batch["image"][(i // n_views) * n_views].eq(batch["image"][i]).all()
        assert (
            batch["sample_idx"][(i // n_views) * n_views]
            .eq(batch["sample_idx"][i])
            .all()
        )
        assert batch["label"][(i // n_views) * n_views].eq(batch["label"][i]).all()


def test_trainer_info():
    import lightning as pl
    import stable_ssl as ossl
    import torch
    from stable_ssl.data import transforms

    train_transform = transforms.ToImage()
    train_dataset = ossl.data.HFDataset(
        path="ylecun/mnist", split="train[:128]", transform=train_transform
    )
    train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=ossl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
        batch_size=64,
        num_workers=20,
        drop_last=True,
    )
    val_transform = transforms.ToImage()
    val = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="ylecun/mnist",
            split="test[:128]",
            transform=val_transform,
        ),
        batch_size=128,
        num_workers=10,
    )
    data = ossl.data.DataModule(train=train, val=val, test=val, predict=val)

    def forward(self, batch, stage):
        assert "sample_idx" in batch
        assert "current_epoch" in batch
        return batch

    module = ossl.Module(dummy=torch.nn.Linear(1, 1), forward=forward, optim=False)
    trainer = pl.Trainer(
        max_epochs=1,
        num_sanity_val_steps=1,
        callbacks=[ossl.callbacks.TrainerInfo()],
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()
    manager.predict()
    manager.test()


if __name__ == "__main__":
    test_repeated_sampler(2)
    test_trainer_info()
