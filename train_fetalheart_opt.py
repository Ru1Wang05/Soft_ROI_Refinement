"""Reproducible entry point for fixed-split D10 fetal-heart experiments."""

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
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_skip", type=int, default=3)
    parser.add_argument("--vit_name", default="R50-ViT-B_16")
    parser.add_argument("--vit_patches_size", type=int, default=16)
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
    parser.add_argument(
        "--tversky_fraction",
        type=float,
        default=0.25,
        help="Tversky share inside the foreground region loss for dice_tversky_ce",
    )
    parser.add_argument("--sampling", choices=["uniform", "class_balanced"], default="uniform")
    parser.add_argument(
        "--augmentation",
        choices=["legacy", "interpolation_fixed", "none"],
        default="legacy",
    )
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_VIT
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from trainer_fetalheart import trainer_fetalheart

    if not torch.cuda.is_available():
        raise RuntimeError("This TransUNet training entry requires a CUDA GPU")
    if abs(args.ce_weight + args.region_weight - 1.0) > 1e-6:
        raise ValueError("ce_weight and region_weight must sum to 1")
    if args.img_size % args.vit_patches_size:
        raise ValueError("img_size must be divisible by vit_patches_size")

    cudnn.benchmark = args.non_deterministic
    cudnn.deterministic = not args.non_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.snapshot_path, exist_ok=True)
    with open(os.path.join(args.snapshot_path, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    config = CONFIGS_VIT[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    config.pretrained_path = args.pretrained_path
    if "R50" in args.vit_name:
        config.patches.grid = (
            args.img_size // args.vit_patches_size,
            args.img_size // args.vit_patches_size,
        )
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    with np.load(args.pretrained_path, allow_pickle=False) as weights:
        model.load_from(weights=weights)
    result = trainer_fetalheart(args, model, args.snapshot_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
