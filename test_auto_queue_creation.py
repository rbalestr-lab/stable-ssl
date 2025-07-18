"""Test that queues are automatically created by evaluation callbacks."""

import torch
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import Accuracy

from stable_ssl.callbacks.knn import OnlineKNN
from stable_ssl.callbacks.rankme import RankMe


class DummyModel(LightningModule):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(784, 768)
        self.classifier = torch.nn.Linear(768, 10)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

    def training_step(self, batch, batch_idx):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))

        features, logits = self(x)

        # Put data in batch dict for callbacks
        batch["features"] = features
        batch["labels"] = y

        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.randn(16, 784)
        y = torch.randint(0, 10, (16,))

        features, logits = self(x)

        batch["features"] = features
        batch["labels"] = y

        return {"loss": torch.nn.functional.cross_entropy(logits, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def test_auto_queue_creation():
    """Test that queues are created automatically."""
    print("Creating model and evaluation callbacks...")
    model = DummyModel()

    # Users ONLY need to create evaluation callbacks
    # Queues will be created automatically!
    callbacks = [
        OnlineKNN(
            name="knn_accuracy",
            input="features",
            target="labels",
            queue_length=1000,
            input_dim=768,
            target_dim=1,
            k=5,
            temperature=0.07,
            metrics={"accuracy": Accuracy(task="multiclass", num_classes=10)},
        ),
        RankMe(
            name="rankme_score", target="features", queue_length=1000, target_shape=768
        ),
    ]

    print(f"\nCreated {len(callbacks)} evaluation callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}: {cb.name}")

    print("\n⚡ Queues will be created automatically during setup!")

    # Create trainer
    trainer = Trainer(
        max_epochs=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    print("\nStarting training...")
    try:
        trainer.fit(model)

        print("\n✅ Success! Queues were created automatically:")

        # Show which queues were created
        from stable_ssl.callbacks.queue import OnlineQueue

        queue_callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, OnlineQueue)
        ]

        print(f"\nAuto-created {len(queue_callbacks)} queues:")
        for cb in queue_callbacks:
            print(
                f"  - OnlineQueue for '{cb.key}' (length={cb.queue_length}, dim={cb.dim}, dtype={cb.dtype})"
            )

        print("\n🎉 Users don't need to manage queues anymore!")
        print("Just create evaluation callbacks and queues are handled automatically.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    test_auto_queue_creation()
