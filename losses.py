"""Losses used by the reproducible fetal-heart optimisation experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class LegacyForegroundDiceLoss(nn.Module):
    """Exact foreground Dice behaviour of the original project.

    The background term is zeroed, but the sum is still divided by all classes.
    Keeping this implementation makes the reference run directly comparable.
    """

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        encoded = _one_hot(target, self.num_classes)
        losses = []
        for class_id in range(1, self.num_classes):
            score = probabilities[:, class_id]
            truth = encoded[:, class_id]
            intersection = torch.sum(score * truth)
            denominator = torch.sum(score.square()) + torch.sum(truth.square())
            losses.append(1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return torch.stack(losses).sum() / self.num_classes


class ForegroundDiceLoss(nn.Module):
    """Mean soft Dice over foreground classes only."""

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)[:, 1:]
        encoded = _one_hot(target, self.num_classes)[:, 1:]
        dimensions = (0, 2, 3)
        intersection = torch.sum(probabilities * encoded, dim=dimensions)
        denominator = torch.sum(probabilities.square(), dim=dimensions) + torch.sum(
            encoded.square(), dim=dimensions
        )
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class ForegroundTverskyLoss(nn.Module):
    """Foreground Tversky loss with an explicit false-negative penalty."""

    def __init__(
        self,
        num_classes: int,
        alpha_fp: float = 0.3,
        beta_fn: float = 0.7,
        gamma: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        if abs(alpha_fp + beta_fn - 1.0) > 1e-6:
            raise ValueError("alpha_fp and beta_fn must sum to 1")
        self.num_classes = num_classes
        self.alpha_fp = alpha_fp
        self.beta_fn = beta_fn
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)[:, 1:]
        encoded = _one_hot(target, self.num_classes)[:, 1:]
        dimensions = (0, 2, 3)
        true_positive = torch.sum(probabilities * encoded, dim=dimensions)
        false_positive = torch.sum(probabilities * (1.0 - encoded), dim=dimensions)
        false_negative = torch.sum((1.0 - probabilities) * encoded, dim=dimensions)
        index = (true_positive + self.smooth) / (
            true_positive
            + self.alpha_fp * false_positive
            + self.beta_fn * false_negative
            + self.smooth
        )
        return torch.pow(1.0 - index, self.gamma).mean()


class HybridDiceTverskyLoss(nn.Module):
    """Conservative blend that keeps Dice while adding an FN-aware term."""

    def __init__(
        self,
        num_classes: int,
        tversky_fraction: float = 0.25,
        alpha_fp: float = 0.3,
        beta_fn: float = 0.7,
        gamma: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= tversky_fraction <= 1.0:
            raise ValueError("tversky_fraction must be between 0 and 1")
        self.tversky_fraction = tversky_fraction
        self.dice = ForegroundDiceLoss(num_classes)
        self.tversky = ForegroundTverskyLoss(
            num_classes,
            alpha_fp=alpha_fp,
            beta_fn=beta_fn,
            gamma=gamma,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.tversky_fraction) * self.dice(
            logits, target
        ) + self.tversky_fraction * self.tversky(logits, target)


class CompositeSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str,
        ce_weight: float = 0.2,
        region_weight: float = 0.8,
        alpha_fp: float = 0.3,
        beta_fn: float = 0.7,
        tversky_gamma: float = 1.0,
        tversky_fraction: float = 0.25,
    ):
        super().__init__()
        self.mode = mode
        self.ce_weight = ce_weight
        self.region_weight = region_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        if mode == "legacy_dice_ce":
            self.region_loss = LegacyForegroundDiceLoss(num_classes)
        elif mode == "dice_ce":
            self.region_loss = ForegroundDiceLoss(num_classes)
        elif mode == "tversky_ce":
            self.region_loss = ForegroundTverskyLoss(
                num_classes,
                alpha_fp=alpha_fp,
                beta_fn=beta_fn,
                gamma=tversky_gamma,
            )
        elif mode == "dice_tversky_ce":
            self.region_loss = HybridDiceTverskyLoss(
                num_classes,
                tversky_fraction=tversky_fraction,
                alpha_fp=alpha_fp,
                beta_fn=beta_fn,
                gamma=tversky_gamma,
            )
        else:
            raise ValueError(f"Unknown loss mode: {mode}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        loss_ce = self.cross_entropy(logits, target.long())
        loss_region = self.region_loss(logits, target)
        total = self.ce_weight * loss_ce + self.region_weight * loss_region
        return total, loss_ce, loss_region
