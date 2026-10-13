"""Local CPU smoke test for the four baseline architectures.

Checks:
1. forward/backward for every arch at its production-relative resolution;
2. output logits are at input resolution with 12 classes;
3. checkpoint save/load round trip preserves outputs;
4. one epoch of the shared trainer loop (unet) on synthetic NPZ data,
   including evaluate_present_case_class, best_dice.pth, and history.json.
Run:  python tests/test_baselines_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from networks.baselines_fetalheart import BASELINE_ARCHS, build_baseline  # noqa: E402

NUM_CLASSES = 12


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
    import shutil

    from datasets.dataset_fetalheart import FetalHeartNPZDataset  # noqa: F401

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

        import argparse

        from trainer_fetalheart import trainer_fetalheart

        args = argparse.Namespace(
            root_path=str(data_dir),
            list_dir=str(list_dir),
            snapshot_path=str(tmp / "run"),
            num_classes=NUM_CLASSES,
            max_epochs=1,
            batch_size=2,
            n_gpu=1,
            base_lr=0.01,
            img_size=img_size,
            seed=42,
            loss_mode="dice_ce",
            ce_weight=0.2,
            region_weight=0.8,
            tversky_alpha_fp=0.3,
            tversky_beta_fn=0.7,
            tversky_gamma=1.0,
            tversky_fraction=0.25,
            sampling="uniform",
            augmentation="none",
            val_every=1,
            num_workers=0,
            amp=False,
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
    # production resolution forward check for swinunet (window divisibility)
    check_arch("swinunet", 384)
    # smaller checks for the CNN decoders (cheaper on CPU)
    check_arch("unet", 128)
    check_arch("attunet", 128)
    check_arch("unetpp", 128)
    check_trainer_loop("unet", 64)
    print("ALL_SMOKE_TESTS_PASSED")
