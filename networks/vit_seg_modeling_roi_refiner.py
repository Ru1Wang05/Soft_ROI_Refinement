"""Soft ROI-guided local refinement for small fetal-heart structures."""
from __future__ import annotations

import torch
import torch.nn as nn

from .vit_seg_modeling import VisionTransformer


class SoftROIRefinerVisionTransformer(VisionTransformer):
    """Shared TransUNet plus a zero-initialized weak-structure residual branch.

    The branch predicts four localization maps and four residual logits. The
    localization maps are soft attention only; no pixels are hard-masked, so a
    failed coarse prediction can fall back to the shared segmentation output.
    """

    WEAK_CLASS_IDS = (8, 9, 10, 11)

    def __init__(self, config, img_size=224, num_classes=12, roi_channels=32,
                 roi_scale=1.0, **kwargs):
        super().__init__(config, img_size=img_size, num_classes=num_classes, **kwargs)
        self.roi_scale = float(roi_scale)
        decoder_channels = config.decoder_channels[-1]
        weak_count = len(self.WEAK_CLASS_IDS)
        self.image_projection = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.roi_localizer = nn.Sequential(
            nn.Conv2d(decoder_channels + 8 + weak_count, roi_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(roi_channels), nn.ReLU(inplace=True),
            nn.Conv2d(roi_channels, weak_count, 1),
        )
        self.roi_residual = nn.Sequential(
            nn.Conv2d(decoder_channels + 8 + weak_count, roi_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(roi_channels), nn.ReLU(inplace=True),
            nn.Conv2d(roi_channels, weak_count, 1),
        )
        nn.init.zeros_(self.roi_residual[-1].weight)
        nn.init.zeros_(self.roi_residual[-1].bias)

    def set_roi_scale(self, value: float):
        self.roi_scale = float(value)

    def forward(self, x, return_aux=False, roi_scale=None):
        if x.size(1) == 1:
            model_input = x.repeat(1, 3, 1, 1)
        else:
            model_input = x
        tokens, attention_weights, features = self.transformer(model_input)
        decoded = self.decoder(tokens, features)
        raw_logits = self.segmentation_head(decoded)
        # Class IDs are already 0-based channel indices for the 12-channel
        # logits (0 = background); the evaluator and CE loss use the same
        # convention. Do NOT subtract 1 here — an earlier version did, which
        # made the branch modify MPA/LPA/RPA/SVC while being supervised on
        # LPA/RPA/SVC/Trachea masks.
        weak_indices = list(self.WEAK_CLASS_IDS)
        coarse_probs = torch.sigmoid(raw_logits[:, weak_indices])
        image_feature = self.image_projection(x[:, :1])
        roi_input = torch.cat([decoded, image_feature, coarse_probs], dim=1)
        roi_attention_logits = self.roi_localizer(roi_input)
        roi_attention = torch.sigmoid(roi_attention_logits)
        residual = self.roi_residual(roi_input) * roi_attention
        scale = self.roi_scale if roi_scale is None else float(roi_scale)
        segmentation_logits = raw_logits.clone()
        segmentation_logits[:, weak_indices] = (
            segmentation_logits[:, weak_indices] + scale * residual
        )
        if not return_aux:
            return segmentation_logits
        return {
            "segmentation": segmentation_logits,
            "raw_segmentation": raw_logits,
            "roi_attention_logits": roi_attention_logits,
            "roi_attention": roi_attention,
            "roi_residual": residual,
            "attention_weights": attention_weights,
        }
