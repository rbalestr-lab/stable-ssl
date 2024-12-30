"""
This example demonstrates how to create a custom supervised model using the
`stable_ssl` library.
"""

import hydra
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from torchvision import transforms

import stable_ssl
from stable_ssl.supervised import Supervised


class MyCustomSupervised(Supervised):
    def initialize_train_loader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=self.config.root, train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.config.optim.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        return trainloader

    def initialize_test_loader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.config.root, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config.optim.batch_size,
            shuffle=False,
            num_workers=2,
        )
        return testloader

    def compute_loss(self):
        """The computer loss is called during training on each mini-batch
        stable-SSL automatically stores the output of the data loader as `self.data`
        which you can access directly within that function"""
        preds = self.forward(self.data[0])
        return F.cross_entropy(preds, self.data[1])


@hydra.main()
def main(cfg: DictConfig):
    args = stable_ssl.get_args(cfg)

    print("--- Arguments ---")
    print(args)

    # while we provide a lot of config parameters (e.g. `optim.batch_size`), you can
    # also pass arguments directly when calling your model, they will be logged and
    # accessible from within the model as `self.config.root` (in this example)
    trainer = MyCustomSupervised(args, root="~/data")
    trainer()


if __name__ == "__main__":
    main()
