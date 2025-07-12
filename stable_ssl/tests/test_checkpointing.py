# def test_sklearn_module_checkpointing_without_callback():

#     import optimalssl as ossl
#     import torch
#     import lightning as pl
#     from omegaconf import OmegaConf
#     import torchvision
#     from torch import nn
#     import sklearn
#     from pathlib import Path
#     import logging
#     import wandb
#     import sklearn.tree

#     def forward(self, batch):
#         return 0

#     wandb.init(mode="disabled")

#     logging.basicConfig(level=logging.INFO)

#     dataset = torch.utils.data.TensorDataset(
#         torch.randn(100, 3, 32, 32), torch.randint(low=0, high=10, size=(100,))
#     )
#     loader = torch.utils.data.DataLoader(dataset, batch_size=10)

#     p = ossl.module.Module(
#         backbone=ossl.backbone.Resnet9(num_classes=10, num_channels=1),
#         linear_probe=nn.Linear(10, 10),
#         nonlinear_probe=torchvision.ops.MLP(
#             in_channels=10,
#             hidden_channels=[2048, 2048, 10],
#             norm_layer=torch.nn.BatchNorm1d,
#         ),
#         tree=sklearn.tree.DecisionTreeRegressor(),
#         G_scaling=1,
#         imbalance_factor=0,
#         batch_size=512,
#         lr=1,
#         forward=forward,
#     )
#     trainer = pl.Trainer(
#         max_epochs=0,
#         accelerator="cpu",
#         enable_checkpointing=False,
#         logger=False,
#         limit_train_batches=2,
#     )

#     trainer.fit(p, train_dataloaders=loader)
#     p.tree.test = 3
#     trainer.save_checkpoint("test.ckpt")
#     p.tree.test = 5
#     trainer.fit(p, train_dataloaders=loader, ckpt_path="test.ckpt")
#     assert p.tree.test == 5
#     del trainer
#     del p

#     p = ossl.module.Module(
#         backbone=ossl.backbone.Resnet9(num_classes=10, num_channels=1),
#         linear_probe=nn.Linear(10, 10),
#         nonlinear_probe=torchvision.ops.MLP(
#             in_channels=10,
#             hidden_channels=[2048, 2048, 10],
#             norm_layer=torch.nn.BatchNorm1d,
#         ),
#         tree=sklearn.tree.DecisionTreeRegressor(),
#         G_scaling=1,
#         imbalance_factor=0,
#         batch_size=512,
#         lr=1,
#         forward=forward,
#     )
#     trainer = pl.Trainer(
#         max_epochs=0,
#         accelerator="cpu",
#         enable_checkpointing=False,
#         logger=False,
#         limit_train_batches=2,
#     )
#     trainer.fit(p, train_dataloaders=loader, ckpt_path="test.ckpt")
#     assert not hasattr(p.tree, "test")
#     Path("test.ckpt").unlink()


# def test_sklearn_module_checkpointing_with_callback():

#     import optimalssl as ossl
#     import torch
#     import lightning as pl
#     from omegaconf import OmegaConf
#     import torchvision
#     from torch import nn
#     import sklearn
#     from pathlib import Path
#     import logging
#     import wandb
#     import sklearn.tree

#     def forward(self, batch):
#         return 0

#     wandb.init(mode="disabled")

#     logging.basicConfig(level=logging.INFO)

#     dataset = torch.utils.data.TensorDataset(
#         torch.randn(100, 3, 32, 32), torch.randint(low=0, high=10, size=(100,))
#     )
#     loader = torch.utils.data.DataLoader(dataset, batch_size=10)

#     p = ossl.module.Module(
#         backbone=ossl.backbone.Resnet9(num_classes=10, num_channels=1),
#         linear_probe=nn.Linear(10, 10),
#         nonlinear_probe=torchvision.ops.MLP(
#             in_channels=10,
#             hidden_channels=[2048, 2048, 10],
#             norm_layer=torch.nn.BatchNorm1d,
#         ),
#         tree=sklearn.tree.DecisionTreeRegressor(),
#         G_scaling=1,
#         imbalance_factor=0,
#         batch_size=512,
#         lr=1,
#         forward=forward,
#     )
#     trainer = pl.Trainer(
#         max_epochs=0,
#         accelerator="cpu",
#         enable_checkpointing=False,
#         logger=False,
#         callbacks=[ossl.callbacks.SklearnCheckpoint()],
#     )

#     trainer.fit(p, train_dataloaders=loader)
#     p.tree.test = 3
#     trainer.save_checkpoint("test.ckpt")
#     p.tree.test = 5
#     trainer.fit(p, train_dataloaders=loader, ckpt_path="test.ckpt")
#     assert p.tree.test == 3
#     del trainer
#     del p

#     p = ossl.module.Module(
#         backbone=ossl.backbone.Resnet9(num_classes=10, num_channels=1),
#         linear_probe=nn.Linear(10, 10),
#         nonlinear_probe=torchvision.ops.MLP(
#             in_channels=10,
#             hidden_channels=[2048, 2048, 10],
#             norm_layer=torch.nn.BatchNorm1d,
#         ),
#         # tree=sklearn.tree.DecisionTreeRegressor(),
#         G_scaling=1,
#         imbalance_factor=0,
#         batch_size=512,
#         lr=1,
#         forward=forward,
#     )
#     trainer = pl.Trainer(
#         max_epochs=0,
#         accelerator="cpu",
#         enable_checkpointing=False,
#         logger=False,
#         callbacks=[ossl.callbacks.SklearnCheckpoint()],
#     )
#     trainer.fit(p, train_dataloaders=loader, ckpt_path="test.ckpt")
#     assert p.tree.test == 3
#     Path("test.ckpt").unlink()


# if __name__ == "__main__":
#     test_sklearn_module_checkpointing_without_callback()
#     test_sklearn_module_checkpointing_with_callback()
