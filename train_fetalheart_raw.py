"""Strict raw TransUNet control: legacy foreground Dice, random initialization."""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def main():
    p = argparse.ArgumentParser(description="Train raw TransUNet without ImageNet-21K weights")
    p.add_argument("--root_path", required=True)
    p.add_argument("--list_dir", required=True)
    p.add_argument("--snapshot_path", required=True)
    p.add_argument("--num_classes", type=int, default=12)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--base_lr", type=float, default=0.01)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_skip", type=int, default=3)
    p.add_argument("--vit_name", default="R50-ViT-B_16")
    p.add_argument("--vit_patches_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Raw TransUNet training requires CUDA")
    if args.img_size % args.vit_patches_size:
        raise ValueError("img_size must be divisible by vit_patches_size")

    from networks.vit_seg_modeling import CONFIGS as CONFIGS_VIT
    from networks.vit_seg_modeling import VisionTransformer
    from trainer_fetalheart import trainer_fetalheart

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.snapshot_path, exist_ok=True)
    with open(os.path.join(args.snapshot_path, "experiment_config.json"), "w", encoding="utf-8") as h:
        json.dump({**vars(args), "loss_mode": "legacy_dice_ce", "pretrained": False,
                   "initialization": "PyTorch random initialization"}, h, indent=2)

    config = CONFIGS_VIT[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    # No call to model.load_from(): all weights remain randomly initialized.
    if "R50" in args.vit_name:
        config.patches.grid = (args.img_size // args.vit_patches_size,
                               args.img_size // args.vit_patches_size)
    model = VisionTransformer(config, img_size=args.img_size,
                              num_classes=args.num_classes).cuda()
    # Preserve the raw/reference training protocol used by the original baseline.
    args.loss_mode = "legacy_dice_ce"
    args.ce_weight = 0.2
    args.region_weight = 0.8
    args.tversky_alpha_fp = 0.3
    args.tversky_beta_fn = 0.7
    args.tversky_gamma = 1.0
    args.tversky_fraction = 0.25
    args.sampling = "uniform"
    args.augmentation = "legacy"
    args.val_every = 2
    result = trainer_fetalheart(args, model, args.snapshot_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
