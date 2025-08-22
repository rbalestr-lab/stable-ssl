"""Common forward functions for use with config-based training.

This module provides pre-defined forward functions that can be used
directly in YAML configs via the _target_ field.
"""

import torch
import stable_pretraining as spt


def simclr_forward(self, batch, stage):
    """Forward function for SimCLR training.

    Args:
        self: Module instance (automatically bound)
        batch: Input batch dictionary
        stage: Training stage (train/val/test)

    Returns:
        Dictionary with loss and embeddings
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        # Project embeddings for contrastive loss
        proj = self.projector(out["embedding"])

        # Fold views for contrastive learning
        views = spt.data.fold_views(proj, batch["sample_idx"])

        # Compute SimCLR loss
        out["loss"] = self.simclr_loss(views[0], views[1])

    return out


def byol_forward(self, batch, stage):
    """Forward function for BYOL training.

    Args:
        self: Module instance (automatically bound)
        batch: Input batch dictionary
        stage: Training stage (train/val/test)

    Returns:
        Dictionary with loss and embeddings
    """
    out = {}

    # Get embeddings from online network
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        # Online projections
        online_proj = self.projector(out["embedding"])
        online_pred = self.predictor(online_proj)

        # Target projections (with momentum encoder)
        with torch.no_grad():
            target_embedding = self.target_backbone(batch["image"])
            target_proj = self.target_projector(target_embedding)

        # Fold views
        online_views = spt.data.fold_views(online_pred, batch["sample_idx"])
        target_views = spt.data.fold_views(target_proj, batch["sample_idx"])

        # BYOL loss (MSE between predictions and targets)
        loss1 = torch.nn.functional.mse_loss(online_views[0], target_views[1].detach())
        loss2 = torch.nn.functional.mse_loss(online_views[1], target_views[0].detach())
        out["loss"] = (loss1 + loss2) / 2

    return out


def vicreg_forward(self, batch, stage):
    """Forward function for VICReg training.

    Args:
        self: Module instance (automatically bound)
        batch: Input batch dictionary
        stage: Training stage (train/val/test)

    Returns:
        Dictionary with loss and embeddings
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        # Project embeddings
        proj = self.projector(out["embedding"])

        # Fold views for VICReg loss
        views = spt.data.fold_views(proj, batch["sample_idx"])

        # Compute VICReg loss (variance + invariance + covariance)
        out["loss"] = self.vicreg_loss(views[0], views[1])

    return out


def barlow_twins_forward(self, batch, stage):
    """Forward function for Barlow Twins training.

    Args:
        self: Module instance (automatically bound)
        batch: Input batch dictionary
        stage: Training stage (train/val/test)

    Returns:
        Dictionary with loss and embeddings
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        # Project embeddings
        proj = self.projector(out["embedding"])

        # Fold views
        views = spt.data.fold_views(proj, batch["sample_idx"])

        # Compute Barlow Twins loss
        out["loss"] = self.barlow_loss(views[0], views[1])

    return out


def supervised_forward(self, batch, stage):
    """Forward function for supervised training.

    Args:
        self: Module instance (automatically bound)
        batch: Input batch dictionary
        stage: Training stage (train/val/test)

    Returns:
        Dictionary with loss and predictions
    """
    out = {}

    # Get embeddings and predictions
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        # Compute cross-entropy loss
        out["loss"] = torch.nn.functional.cross_entropy(out["logits"], batch["label"])

    return out
