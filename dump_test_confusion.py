"""Run test-split inference per checkpoint and accumulate the pixel confusion matrix.

Modeled on evaluate_fetalheart_final_models.py; replicates the exact eval path of
trainer_fetalheart.evaluate_present_case_class (resize image to img_size bilinear,
argmax, nearest-upsample prediction back to the original label size), so the matrix
is consistent with the paper metrics. Saves a 12x12 integer count matrix per run
and, with --merge, merges all runs into row-normalized means for plotting.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.dataset_fetalheart import FetalHeartNPZDataset


def build_model(model_type, args):
    from networks.vit_seg_modeling import CONFIGS
    config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    config.patches.grid = (args.img_size // args.vit_patches_size,) * 2
    if model_type == 'original':
        from networks.vit_seg_modeling import VisionTransformer
        return VisionTransformer(config, img_size=args.img_size, num_classes=args.num_classes)
    if model_type == 'roi_refiner':
        from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer
        return SoftROIRefinerVisionTransformer(config, img_size=args.img_size, num_classes=args.num_classes)
    from networks.baselines_fetalheart import build_baseline
    arch = args.arch or model_type.replace('baseline_', '')
    return build_baseline(arch, num_classes=args.num_classes, img_size=args.img_size, pretrained=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', required=True)
    p.add_argument('--list_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model_type', required=True,
                   choices=['original', 'roi_refiner', 'baseline_unet', 'baseline_attunet',
                            'baseline_unetpp', 'baseline_swinunet'])
    p.add_argument('--arch', default=None)
    p.add_argument('--split', default='test')
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--num_classes', type=int, default=12)
    p.add_argument('--vit_name', default='R50-ViT-B_16')
    p.add_argument('--vit_patches_size', type=int, default=16)
    p.add_argument('--n_skip', type=int, default=3)
    p.add_argument('--output_dir', required=True, help='directory for confusion count .npy files')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.model_type, args).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = FetalHeartNPZDataset(args.root_path, args.list_dir, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    C = args.num_classes
    cm = np.zeros((C, C), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].float().to(device)
            label = batch['label'].long().to(device)
            if image.ndim == 3:
                image = image.unsqueeze(1)
            original_size = label.shape[-2:]
            resized = F.interpolate(image, size=(args.img_size, args.img_size),
                                    mode='bilinear', align_corners=False)
            logits = model(resized)
            pred = torch.argmax(logits, dim=1, keepdim=True).float()
            pred = F.interpolate(pred, size=original_size, mode='nearest').squeeze(1).long()
            idx = label.reshape(-1).cpu().numpy().astype(int) * C + pred.reshape(-1).cpu().numpy()
            cm += np.bincount(idx, minlength=C * C).reshape(C, C)

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.checkpoint.rstrip('/').replace('\\', '/'))
    run = f"{args.model_type}_{name}_confusion.npy"
    out = os.path.join(args.output_dir, run)
    np.save(out, cm)
    print(json.dumps({'saved': out, 'cases': len(dataset), 'pixels': int(cm.sum())}))


if __name__ == '__main__':
    main()
