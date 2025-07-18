"""Test script to demonstrate the new queue discovery architecture."""

import torch
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import Accuracy

from stable_ssl.callbacks.knn import OnlineKNN

# Import the new callbacks
from stable_ssl.callbacks.queue import OnlineQueue
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
        # Simulate a batch with features and labels
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))

        features, logits = self(x)

        # Put data in batch dict for callbacks
        batch["features"] = features
        batch["labels"] = y

        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # Same as training but callbacks will use the queued data
        x = torch.randn(16, 784)
        y = torch.randint(0, 10, (16,))

        features, logits = self(x)

        batch["features"] = features
        batch["labels"] = y

        return {"loss": torch.nn.functional.cross_entropy(logits, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def test_new_architecture():
    """Test the new queue discovery architecture."""
    print("Creating model and callbacks...")
    model = DummyModel()

    # Step 1: Create queue callbacks for data collection
    # Each queue tracks ONE piece of data
    queue_callbacks = [
        OnlineQueue(
            key="features",
            queue_length=1000,
            dim=768,
            dtype=torch.float32,
            gather_distributed=False,
        ),
        OnlineQueue(
            key="labels",
            queue_length=1000,
            dim=1,
            dtype=torch.long,
            gather_distributed=False,
        ),
    ]

    # Step 2: Create evaluation callbacks that automatically discover queues
    eval_callbacks = [
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

    # Combine all callbacks
    all_callbacks = queue_callbacks + eval_callbacks

    print(f"\nCreated {len(queue_callbacks)} queue callbacks:")
    for cb in queue_callbacks:
        print(
            f"  - OnlineQueue for '{cb.key}' (length={cb.queue_length}, dim={cb.dim}, dtype={cb.dtype})"
        )

    print(f"\nCreated {len(eval_callbacks)} evaluation callbacks:")
    for cb in eval_callbacks:
        print(f"  - {cb.__class__.__name__}: {cb.name}")

    # Create trainer with callbacks
    trainer = Trainer(
        max_epochs=1,
        callbacks=all_callbacks,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    print("\nStarting training...")
    try:
        # This will trigger the queue discovery during setup
        trainer.fit(model)
        print("\n✅ Success! The new architecture works correctly.")

        print("\nKey benefits of the new architecture:")
        print("1. Clean separation: Queues collect data, evaluation callbacks use it")
        print("2. Automatic discovery: Callbacks find queues by data key + properties")
        print("3. No complex sharing logic: Each queue is independent")
        print("4. Type safety: Mismatched properties are caught early")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    test_new_architecture()
