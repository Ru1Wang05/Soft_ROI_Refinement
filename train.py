"""Unified training entry for all models on the fixed D10 fetal-heart split.

Covers the whole experiment progression with one interface:

  --model transunet    official raw (add --loss_mode legacy_dice_ce, no pretrained),
                       factorial controls (loss / pretraining single factors),
                       and the Adjusted TransUNet baseline (384, 70 epochs)
  --model roi_refiner  Adjusted TransUNet + Soft ROI Refiner (proposed)
  --model unet|attunet|unetpp|swinunet   matched CNN/Swin baselines
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

BASELINE_ARCHS = ("unet", "attunet", "unetpp", "swinunet")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("transunet", "roi_refiner") + BASELINE_ARCHS)
    parser.add_argument("--root_path", required=True, help="Directory containing all split NPZ files")
    parser.add_argument("--list_dir", required=True, help="Directory containing fixed train/val/test.txt")
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument("--pretrained_path", default="",
                        help="ImageNet-21k ViT weights (transunet / roi_refiner)")
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
    parser.add_argument("--loss_mode",
                        choices=["legacy_dice_ce", "dice_ce", "tversky_ce", "dice_tversky_ce"],
                        default="dice_ce")
    parser.add_argument("--ce_weight", type=float, default=0.2)
    parser.add_argument("--region_weight", type=float, default=0.8)
    parser.add_argument("--tversky_alpha_fp", type=float, default=0.3)
    parser.add_argument("--tversky_beta_fn", type=float, default=0.7)
    parser.add_argument("--tversky_gamma", type=float, default=1.0)
    parser.add_argument("--tversky_fraction", type=float, default=0.25,
                        help="Tversky share inside the foreground region loss for dice_tversky_ce")
    parser.add_argument("--sampling", choices=["uniform", "class_balanced"], default="uniform")
    parser.add_argument("--augmentation",
                        choices=["legacy", "interpolation_fixed", "none"], default="legacy")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")
    # ROI Refiner only
    parser.add_argument("--roi_loss_weight", type=float, default=0.2)
    parser.add_argument("--roi_scale", type=float, default=1.0)
    parser.add_argument("--roi_pos_weights", default="2,2,2,2")
    # CNN/Swin baselines only
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet initialization (CNN/Swin encoders)")
    parser.add_argument("--swin_pretrained", default=None,
                        help="Optional official Swin-T ImageNet checkpoint for swinunet")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd",
                        help="Optimizer; AdamW follows the original Swin recipe (swinunet)")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.05)
    return parser


def main():
    args = build_parser().parse_args()

    if args.model in ("transunet", "roi_refiner") and not torch.cuda.is_available():
        raise RuntimeError("TransUNet / ROI Refiner training requires a CUDA GPU")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU visible; running on CPU (smoke tests only)")
    if abs(args.ce_weight + args.region_weight - 1.0) > 1e-6:
        raise ValueError("ce_weight and region_weight must sum to 1")
    if args.pretrained_path and not os.path.isfile(args.pretrained_path):
        raise FileNotFoundError(args.pretrained_path)

    if args.model == "swinunet" and args.optimizer == "sgd" and args.warmup_epochs == 0:
        # The shared SGD schedule does not train the pure-transformer Swin-Unet
        # (loss plateaus near 0.47, val Dice stays below 0.40 within 70 epochs);
        # fall back to its original AdamW recipe with linear warmup. The
        # effective settings are recorded in experiment_config.json.
        args.optimizer = "adamw"
        args.warmup_epochs = 5
        args.base_lr = 5e-4
        print("swinunet: overriding optimizer to AdamW lr=5e-4, weight_decay=0.05, warmup_epochs=5")

    if args.model in ("transunet", "roi_refiner"):
        if args.img_size % args.vit_patches_size:
            raise ValueError("img_size must be divisible by vit_patches_size")
        from networks.vit_seg_modeling import CONFIGS, VisionTransformer
        config = CONFIGS[args.vit_name]
        config.n_classes = args.num_classes
        config.n_skip = args.n_skip
        if "R50" in args.vit_name:
            config.patches.grid = (args.img_size // args.vit_patches_size,) * 2
        if args.model == "roi_refiner":
            from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer
            args.roi_pos_weights = [float(item) for item in args.roi_pos_weights.split(",")]
            if len(args.roi_pos_weights) != 4:
                raise ValueError("roi_pos_weights must contain four comma-separated values")
            model = SoftROIRefinerVisionTransformer(config, img_size=args.img_size,
                                                     num_classes=args.num_classes,
                                                     roi_scale=args.roi_scale).cuda()
        else:
            model = VisionTransformer(config, img_size=args.img_size,
                                      num_classes=args.num_classes).cuda()
        if args.pretrained_path:
            with np.load(args.pretrained_path, allow_pickle=False) as weights:
                model.load_from(weights=weights)
    else:
        from networks.baselines_fetalheart import build_baseline
        model = build_baseline(args.model, num_classes=args.num_classes, img_size=args.img_size,
                               pretrained=not args.no_pretrained,
                               swin_pretrained=args.swin_pretrained)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = args.non_deterministic
    cudnn.deterministic = not args.non_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.snapshot_path, exist_ok=True)
    config_record = {**vars(args),
                     "pretrained": bool(args.pretrained_path) if args.model in ("transunet", "roi_refiner")
                     else not args.no_pretrained,
                     "initialization": "ImageNet-21K" if args.pretrained_path
                     or (args.model in BASELINE_ARCHS and not args.no_pretrained)
                     else "PyTorch random initialization"}
    with open(os.path.join(args.snapshot_path, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_record, handle, indent=2, ensure_ascii=False)

    from trainers import trainer_fetalheart, trainer_fetalheart_roi_refiner
    if args.model == "roi_refiner":
        result = trainer_fetalheart_roi_refiner(args, model, args.snapshot_path)
    else:
        result = trainer_fetalheart(args, model, args.snapshot_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
