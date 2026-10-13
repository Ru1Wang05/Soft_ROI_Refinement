"""Self-contained CPU checks for the whole pipeline.

Sections:
1. losses / evaluation protocol invariants (corrected Dice, missed-structure-zero)
2. ROI Refiner channel mapping (off-by-one regression guard)
3. baseline architecture smoke tests + one epoch of the shared trainer loop

Run:  python tests/run_checks.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.dataset_fetalheart import FetalHeartTrainTransform  # noqa: E402
from losses import ForegroundDiceLoss, HybridDiceTverskyLoss, LegacyForegroundDiceLoss  # noqa: E402
from networks.baselines_fetalheart import build_baseline  # noqa: E402
from trainers import evaluate_present_case_class, trainer_fetalheart  # noqa: E402

NUM_CLASSES = 12


# ---------- 1. losses and evaluation protocol ----------

def check_corrected_dice_averages_over_foreground_only():
    torch.manual_seed(7)
    logits = torch.randn(2, 12, 8, 8)
    target = torch.randint(0, 12, (2, 8, 8))
    legacy = LegacyForegroundDiceLoss(12)(logits, target)
    corrected = ForegroundDiceLoss(12)(logits, target)
    assert torch.allclose(legacy, corrected * 11.0 / 12.0, atol=1e-6)


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


def check_missed_present_structure_scores_zero():
    metrics = evaluate_present_case_class(
        _AlwaysBackground(), DataLoader(_OneMissedStructure(), batch_size=1), 2, 8, "cpu"
    )
    assert metrics["present_case_class_count"] == 1
    assert metrics["mean_dice"] == 0.0
    assert metrics["mean_iou"] == 0.0


def check_resize_preserves_integer_labels():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    label = np.zeros((8, 8), dtype=np.int64)
    label[2:6, 2:6] = 3
    transformed = FetalHeartTrainTransform([13, 13], augmentation="none")({"image": image, "label": label})
    assert tuple(transformed["image"].shape) == (1, 13, 13)
    assert tuple(transformed["label"].shape) == (13, 13)
    assert set(transformed["label"].unique().tolist()) == {0, 3}


def check_hybrid_loss_endpoints_match_components():
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    dice = ForegroundDiceLoss(4)(logits, target)
    hybrid_dice = HybridDiceTverskyLoss(4, tversky_fraction=0.0)(logits, target)
    assert torch.allclose(dice, hybrid_dice, atol=1e-6)


# ---------- 2. ROI Refiner channel mapping ----------

def check_roi_refiner_channel_mapping():
    """The refiner must modify exactly the channels of its WEAK_CLASS_IDS.

    Regression for the off-by-one audit finding: an earlier forward pass
    indexed logits with class_id - 1, so the branch was supervised on
    LPA/RPA/SVC/Trachea masks while modifying MPA/LPA/RPA/SVC logits.
    """
    from networks.vit_seg_modeling import CONFIGS
    from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer

    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = 12
    config.n_skip = 3
    config.patches.grid = (14, 14)
    model = SoftROIRefinerVisionTransformer(config, img_size=224, num_classes=12)
    model.eval()

    weak = list(model.WEAK_CLASS_IDS)
    with torch.no_grad():
        probe = model(torch.rand(1, 1, 224, 224), return_aux=True)
    big = float(probe["raw_segmentation"].abs().max()) * 2.0 + 10.0
    res_hw = (224, 224)  # the branch operates at full input resolution
    for j, expected_channel in enumerate(weak):
        synthetic = torch.zeros(1, 4, *res_hw)
        synthetic[0, j, 32, 48] = big
        synthetic[0, j, 80, 96] = -big

        def fake_localizer(module, inputs, output):
            return output.fill_(0.0)  # sigmoid -> attention exactly 1

        def fake_residual(module, inputs, output):
            return synthetic.clone()

        h1 = model.roi_localizer.register_forward_hook(fake_localizer)
        h2 = model.roi_residual.register_forward_hook(fake_residual)
        try:
            with torch.no_grad():
                out = model(torch.rand(1, 1, 224, 224), return_aux=True)
        finally:
            h1.remove()
            h2.remove()

        pred = out["segmentation"].argmax(1)[0]
        # where the synthetic residual is +big the argmax must be the wired class
        assert pred[32, 48].item() == expected_channel, (
            f"residual channel {j} drives logits channel {pred[32, 48].item()}, "
            f"expected {expected_channel} (off-by-one regression?)"
        )
        # where it is -big the wired class must be suppressed if raw predicted it,
        # so just require it is not forced to the wired class
        assert pred[80, 96].item() != expected_channel or out["raw_segmentation"][0, expected_channel, 80, 96] > 0


# ---------- 3. baseline architectures + shared trainer loop ----------

def check_arch(arch: str, img_size: int) -> None:
    print(f"[{arch}] building (img_size={img_size}) ...", flush=True)
    model = build_baseline(arch, num_classes=NUM_CLASSES, img_size=img_size, pretrained=False)
    model.train()
    x = torch.randn(2, 1, img_size, img_size)
    logits = model(x)
    assert logits.shape == (2, NUM_CLASSES, img_size, img_size), logits.shape
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, NUM_CLASSES, (2, img_size, img_size)))
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        ref = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pth"
        torch.save(model.state_dict(), path)
        clone = build_baseline(arch, num_classes=NUM_CLASSES, img_size=img_size, pretrained=False)
        clone.load_state_dict(torch.load(path, map_location="cpu"))
        clone.eval()
        with torch.no_grad():
            out = clone(x)
    assert torch.equal(ref, out), f"{arch}: checkpoint round trip changed outputs"
    print(f"[{arch}] OK  params={n_params/1e6:.2f}M  logits={tuple(logits.shape)}", flush=True)


def check_trainer_loop(arch: str = "unet", img_size: int = 64) -> None:
    """Run the shared trainer for one epoch on synthetic NPZ data (CPU)."""
    import argparse

    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="bl_smoke_"))
    try:
        data_dir = tmp / "npz"
        list_dir = tmp / "lists"
        data_dir.mkdir()
        list_dir.mkdir()
        for split, count in (("train", 6), ("val", 3)):
            names = []
            for i in range(count):
                name = f"{split}_{i:03d}"
                names.append(name)
                label = rng.integers(0, NUM_CLASSES, size=(img_size, img_size))
                # guarantee every foreground class appears somewhere
                for c in range(1, NUM_CLASSES):
                    label[c, c] = c
                image = rng.integers(0, 255, size=(img_size, img_size)).astype(np.float32)
                np.savez(data_dir / f"{name}.npz", image=image, label=label.astype(np.int64))
            (list_dir / f"{split}.txt").write_text("\n".join(names), encoding="utf-8")

        args = argparse.Namespace(
            root_path=str(data_dir), list_dir=str(list_dir), snapshot_path=str(tmp / "run"),
            num_classes=NUM_CLASSES, max_epochs=1, batch_size=2, n_gpu=1, base_lr=0.01,
            img_size=img_size, seed=42, loss_mode="dice_ce", ce_weight=0.2, region_weight=0.8,
            tversky_alpha_fp=0.3, tversky_beta_fn=0.7, tversky_gamma=1.0, tversky_fraction=0.25,
            sampling="uniform", augmentation="none", val_every=1, num_workers=0, amp=False,
        )
        model = build_baseline(arch, num_classes=NUM_CLASSES, img_size=img_size, pretrained=False)
        trainer_fetalheart(args, model, args.snapshot_path)
        for artifact in ("best_dice.pth", "last.pth", "best_metrics.json", "history.json"):
            assert (Path(args.snapshot_path) / artifact).is_file(), f"missing {artifact}"
        print(f"[trainer:{arch}] OK  snapshot={args.snapshot_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    check_corrected_dice_averages_over_foreground_only()
    check_missed_present_structure_scores_zero()
    check_resize_preserves_integer_labels()
    check_hybrid_loss_endpoints_match_components()
    print("[losses/protocol] OK")

    check_roi_refiner_channel_mapping()
    print("[roi_refiner] OK")

    # production resolution forward check for swinunet (window divisibility)
    check_arch("swinunet", 384)
    # smaller checks for the CNN decoders (cheaper on CPU)
    check_arch("unet", 128)
    check_arch("attunet", 128)
    check_arch("unetpp", 128)
    check_trainer_loop("unet", 64)
    print("ALL_CHECKS_PASSED")
