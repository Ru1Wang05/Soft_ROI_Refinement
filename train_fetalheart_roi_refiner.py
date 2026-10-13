"""Entry point for soft ROI refiner seed 42."""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True); parser.add_argument("--list_dir", required=True)
    parser.add_argument("--snapshot_path", required=True); parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--num_classes", type=int, default=12); parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4); parser.add_argument("--max_epochs", type=int, default=70)
    parser.add_argument("--base_lr", type=float, default=0.01); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1); parser.add_argument("--n_skip", type=int, default=3)
    parser.add_argument("--vit_name", default="R50-ViT-B_16"); parser.add_argument("--vit_patches_size", type=int, default=16)
    parser.add_argument("--augmentation", default="legacy"); parser.add_argument("--roi_loss_weight", type=float, default=0.2)
    parser.add_argument("--roi_scale", type=float, default=1.0)
    parser.add_argument("--roi_pos_weights", default="2,2,2,2")
    parser.add_argument("--val_every", type=int, default=2); parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    from networks.vit_seg_modeling import CONFIGS
    from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer
    from trainer_fetalheart_roi_refiner import trainer_fetalheart_roi_refiner
    cudnn.benchmark = False; cudnn.deterministic = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.snapshot_path, exist_ok=True)
    with open(os.path.join(args.snapshot_path, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    config = CONFIGS[args.vit_name]; config.n_classes = args.num_classes; config.n_skip = args.n_skip
    config.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)
    model = SoftROIRefinerVisionTransformer(config, img_size=args.img_size, num_classes=args.num_classes,
                                            roi_scale=args.roi_scale).cuda()
    with np.load(args.pretrained_path, allow_pickle=False) as weights: model.load_from(weights)
    args.roi_pos_weights = [float(item) for item in args.roi_pos_weights.split(",")]
    if len(args.roi_pos_weights) != 4:
        raise ValueError("roi_pos_weights must contain four comma-separated values")
    print(json.dumps(trainer_fetalheart_roi_refiner(args, model, args.snapshot_path), indent=2))


if __name__ == "__main__": main()
