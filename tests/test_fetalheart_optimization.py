import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_fetalheart import FetalHeartTrainTransform  # noqa: E402
from losses_fetalheart import (  # noqa: E402
    ForegroundDiceLoss,
    HybridDiceTverskyLoss,
    LegacyForegroundDiceLoss,
)
from trainer_fetalheart import evaluate_present_case_class  # noqa: E402
from prepare_fetalheart_view_metadata import case_name_from_image_id  # noqa: E402


class _OneMissedStructure(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        label = torch.zeros((8, 8), dtype=torch.long)
        label[2:6, 2:6] = 1
        return {"image": torch.zeros((8, 8)), "label": label, "case_name": "case"}


class _AlwaysBackground(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        logits = torch.zeros((image.shape[0], 2, image.shape[2], image.shape[3]))
        logits[:, 0] = 1.0 + self.anchor
        return logits


class FetalHeartOptimisationTests(unittest.TestCase):
    def test_corrected_dice_averages_over_foreground_only(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 12, 8, 8)
        target = torch.randint(0, 12, (2, 8, 8))
        legacy = LegacyForegroundDiceLoss(12)(logits, target)
        corrected = ForegroundDiceLoss(12)(logits, target)
        self.assertTrue(torch.allclose(legacy, corrected * 11.0 / 12.0, atol=1e-6))

    def test_missed_present_structure_scores_zero(self):
        metrics = evaluate_present_case_class(
            _AlwaysBackground(), DataLoader(_OneMissedStructure(), batch_size=1), 2, 8, "cpu"
        )
        self.assertEqual(metrics["present_case_class_count"], 1)
        self.assertEqual(metrics["mean_dice"], 0.0)
        self.assertEqual(metrics["mean_iou"], 0.0)

    def test_resize_preserves_integer_labels(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        label = np.zeros((8, 8), dtype=np.int64)
        label[2:6, 2:6] = 3
        transformed = FetalHeartTrainTransform([13, 13], augmentation="none")(
            {"image": image, "label": label}
        )
        self.assertEqual(tuple(transformed["image"].shape), (1, 13, 13))
        self.assertEqual(tuple(transformed["label"].shape), (13, 13))
        self.assertEqual(set(transformed["label"].unique().tolist()), {0, 3})

    def test_hybrid_loss_endpoints_match_components(self):
        torch.manual_seed(11)
        logits = torch.randn(2, 4, 8, 8)
        target = torch.randint(0, 4, (2, 8, 8))
        dice = ForegroundDiceLoss(4)(logits, target)
        hybrid_dice = HybridDiceTverskyLoss(4, tversky_fraction=0.0)(logits, target)
        self.assertTrue(torch.allclose(dice, hybrid_dice, atol=1e-6))

    def test_view_metadata_case_name_matches_npz_convention(self):
        self.assertEqual(case_name_from_image_id("B01/100_image.png"), "B01_100_image")


if __name__ == "__main__":
    unittest.main()
