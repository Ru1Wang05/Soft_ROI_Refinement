"""Reproducible entry point for matched baseline comparisons on fixed-split D10.

Mirrors train_fetalheart_opt.py so that U-Net / Attention U-Net / UNet++ /
Swin-Unet train under the identical protocol as Adjusted TransUNet and the
Soft ROI Refiner: same data loader, loss (corrected foreground Dice + CE),
SGD schedule, augmentation, validation selection, seeds, and 384x384 input.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="Directory containing all split NPZ files")
    parser.add_argument("--list_dir", required=True, help="Directory containing fixed train/val/test.txt")
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument(
        "--arch",
        choices=["unet", "attunet", "unetpp", "swinunet"],
        required=True,
        help="Baseline architecture to train",
    )
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss_mode",
        choices=["legacy_dice_ce", "dice_ce", "tversky_ce", "dice_tversky_ce"],
        default="dice_ce",
    )
    parser.add_argument("--ce_weight", type=float, default=0.2)
    parser.add_argument("--region_weight", type=float, default=0.8)
    parser.add_argument("--tversky_alpha_fp", type=float, default=0.3)
    parser.add_argument("--tversky_beta_fn", type=float, default=0.7)
    parser.add_argument("--tversky_gamma", type=float, default=1.0)
    parser.add_argument("--tversky_fraction", type=float, default=0.25)
    parser.add_argument("--sampling", choices=["uniform", "class_balanced"], default="uniform")
    parser.add_argument(
        "--augmentation",
        choices=["legacy", "interpolation_fixed", "none"],
        default="legacy",
    )
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw"],
        default="sgd",
        help="Optimizer; AdamW follows the original Swin recipe (swinunet)",
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet initialization (CNN encoders / Swin encoder)",
    )
    parser.add_argument(
        "--swin_pretrained",
        default=None,
        help="Optional path to an official Swin-T ImageNet checkpoint for swinunet",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.arch == "swinunet" and args.optimizer == "sgd" and args.warmup_epochs == 0:
        # The shared SGD schedule does not train the pure-transformer Swin-Unet
        # (loss plateaus near 0.47, val Dice stays below 0.40 within 70 epochs);
        # fall back to its original AdamW recipe with linear warmup. The
        # effective settings are recorded in experiment_config.json.
        args.optimizer = "adamw"
        args.warmup_epochs = 5
        args.base_lr = 5e-4
        print("swinunet: overriding optimizer to AdamW lr=5e-4, weight_decay=0.05, warmup_epochs=5")

    from networks.baselines_fetalheart import build_baseline
    from trainer_fetalheart import trainer_fetalheart

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU visible; running on CPU (smoke tests only)")
    if abs(args.ce_weight + args.region_weight - 1.0) > 1e-6:
        raise ValueError("ce_weight and region_weight must sum to 1")

    cudnn.benchmark = args.non_deterministic
    cudnn.deterministic = not args.non_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.snapshot_path, exist_ok=True)
    with open(os.path.join(args.snapshot_path, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    model = build_baseline(
        args.arch,
        num_classes=args.num_classes,
        img_size=args.img_size,
        pretrained=not args.no_pretrained,
        swin_pretrained=args.swin_pretrained,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    result = trainer_fetalheart(args, model, args.snapshot_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
