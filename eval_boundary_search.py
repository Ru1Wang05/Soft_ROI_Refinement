import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, zoom


CLASS_NAMES = {
    1: "Ascending Aorta",
    2: "Descending Aorta",
    3: "Left Atrium",
    4: "Left Ventricle",
    5: "Right Atrium",
    6: "Right Ventricle",
    7: "Main Pulmonary Artery",
    8: "Left Pulmonary Artery",
    9: "Right Pulmonary Artery",
    10: "Superior Vena Cava",
    11: "Trachea",
}


def tag_float(value):
    return str(value).replace(".", "p")


def finite_mean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(valid)) if valid else None


def metric_value(value):
    if value is None or math.isnan(value):
        return ""
    return float(value)


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou_score(pred, gt):
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def make_boundary(mask, radius):
    if not mask.any():
        return mask.astype(bool)
    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    return np.logical_xor(binary_dilation(mask, structure=structure), binary_erosion(mask, structure=structure))


def boundary_dice_score(pred, gt, radius):
    return dice_score(make_boundary(pred, radius), make_boundary(gt, radius))


def surface_distances(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float64)
    if not pred.any() or not gt.any():
        return np.array([np.nan], dtype=np.float64)
    structure = np.ones((3, 3), dtype=bool)
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=structure))
    gt_surface = np.logical_xor(gt, binary_erosion(gt, structure=structure))
    if not pred_surface.any() or not gt_surface.any():
        return np.array([np.nan], dtype=np.float64)
    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    return np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64)


def hd95_score(pred, gt):
    distances = surface_distances(pred, gt)
    if np.isnan(distances).any():
        return None
    return float(np.percentile(distances, 95))


def assd_score(pred, gt):
    distances = surface_distances(pred, gt)
    if np.isnan(distances).any():
        return None
    return float(np.mean(distances))


def load_case(data_dir, case_name, img_size):
    data = np.load(Path(data_dir) / (case_name + ".npz"))
    image = data["image"].astype(np.float32)
    label = data["label"].astype(np.int64)
    original_shape = image.shape
    if image.shape != (img_size, img_size):
        image_for_net = zoom(image, (img_size / image.shape[0], img_size / image.shape[1]), order=3)
    else:
        image_for_net = image
    return image, image_for_net, label, original_shape


def predict_case(model, image_for_net, original_shape):
    tensor = torch.from_numpy(image_for_net).unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    if pred.shape != original_shape:
        pred = zoom(pred, (original_shape[0] / pred.shape[0], original_shape[1] / pred.shape[1]), order=0)
    return pred.astype(np.int64)


def build_model(args):
    if args.model_type.startswith("baseline_"):
        from networks.baselines_fetalheart import build_baseline

        arch = args.model_type.replace("baseline_", "")
        model = build_baseline(
            arch, num_classes=args.num_classes, img_size=args.img_size, pretrained=False
        ).cuda()
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size),
                                   int(args.img_size / args.vit_patches_size))
    if args.model_type == "roi_refiner":
        from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer as ModelClass
    else:
        from networks.vit_seg_modeling import VisionTransformer as ModelClass
    model = ModelClass(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(args):
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation directory: {}".format(out_dir))
    out_dir.mkdir(parents=True)
    labelmap_dir = out_dir / "labelmaps"
    if args.save_labelmaps:
        labelmap_dir.mkdir()

    with open(Path(args.list_dir) / (args.split + ".txt"), "r", encoding="utf-8") as f:
        cases = [line.strip() for line in f if line.strip()]

    print("device: cuda")
    print("cases: {}".format(len(cases)))
    model = build_model(args)
    case_rows = []
    per_class = {class_id: [] for class_id in range(1, args.num_classes)}

    for idx, case_name in enumerate(cases, start=1):
        _, image_for_net, label, original_shape = load_case(args.data_dir, case_name, args.img_size)
        pred = predict_case(model, image_for_net, original_shape)
        if args.save_labelmaps:
            np.save(labelmap_dir / (case_name + "_pred.npy"), pred)
        for class_id in range(1, args.num_classes):
            gt_mask = label == class_id
            pred_mask = pred == class_id

            if not gt_mask.any() and not pred_mask.any():
                continue

            row = {
                "case": case_name,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, "class_{}".format(class_id)),
                "dice": dice_score(pred_mask, gt_mask),
                "iou": iou_score(pred_mask, gt_mask),
                "boundary_dice": boundary_dice_score(pred_mask, gt_mask, args.boundary_metric_radius),
                "hd95": hd95_score(pred_mask, gt_mask),
                "assd": assd_score(pred_mask, gt_mask),
                "gt_pixels": int(gt_mask.sum()),
                "pred_pixels": int(pred_mask.sum()),
            }
            case_rows.append(row)
            per_class[class_id].append(row)
        if idx % 20 == 0 or idx == len(cases):
            print("processed {}/{}".format(idx, len(cases)))

    summary_rows = []
    for class_id in range(1, args.num_classes):
        rows = per_class[class_id]
        summary = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, "class_{}".format(class_id)),
            "n_cases": len(rows),
            "mean_dice": finite_mean([r["dice"] for r in rows]),
            "mean_iou": finite_mean([r["iou"] for r in rows]),
            "mean_boundary_dice": finite_mean([r["boundary_dice"] for r in rows]),
            "mean_hd95": finite_mean([r["hd95"] for r in rows]),
            "mean_assd": finite_mean([r["assd"] for r in rows]),
        }
        summary_rows.append(summary)

    macro = {
        "macro_dice": finite_mean([r["mean_dice"] for r in summary_rows]),
        "macro_iou": finite_mean([r["mean_iou"] for r in summary_rows]),
        "macro_boundary_dice": finite_mean([r["mean_boundary_dice"] for r in summary_rows]),
        "macro_hd95": finite_mean([r["mean_hd95"] for r in summary_rows]),
        "macro_assd": finite_mean([r["mean_assd"] for r in summary_rows]),
    }

    with open(out_dir / "case_metrics.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["case", "class_id", "class_name", "dice", "iou", "boundary_dice", "hd95", "assd",
                      "gt_pixels", "pred_pixels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({k: metric_value(row[k]) if k in {"dice", "iou", "boundary_dice", "hd95", "assd"} else row[k]
                             for k in fieldnames})

    with open(out_dir / "summary_metrics.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["class_id", "class_name", "n_cases", "mean_dice", "mean_iou", "mean_boundary_dice",
                      "mean_hd95", "mean_assd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: metric_value(row[k]) if k.startswith("mean_") else row[k] for k in fieldnames})

    expected = {}
    if args.expect_macro_dice is not None:
        expected["macro_dice"] = {
            "expected": args.expect_macro_dice,
            "actual": macro["macro_dice"],
            "abs_diff": abs(macro["macro_dice"] - args.expect_macro_dice),
            "passed": abs(macro["macro_dice"] - args.expect_macro_dice) <= args.expect_tolerance,
        }
    if args.expect_macro_iou is not None:
        expected["macro_iou"] = {
            "expected": args.expect_macro_iou,
            "actual": macro["macro_iou"],
            "abs_diff": abs(macro["macro_iou"] - args.expect_macro_iou),
            "passed": abs(macro["macro_iou"] - args.expect_macro_iou) <= args.expect_tolerance,
        }

    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "data_dir": args.data_dir,
        "list_dir": args.list_dir,
        "num_cases": len(cases),
        "seed": args.seed,
        "boundary_weight": args.boundary_weight,
        "boundary_radius": args.boundary_radius,
        "boundary_metric_radius": args.boundary_metric_radius,
        "macro": macro,
        "per_class": summary_rows,
        "expected_checks": expected,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("saved: {}".format(out_dir.as_posix()))
    print("summary:")
    for row in summary_rows:
        print(row)
    if expected and not all(check["passed"] for check in expected.values()):
        raise AssertionError("Expected Original metrics did not reproduce: {}".format(expected))


def self_test():
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:10, 4:10] = True
    perfect = gt.copy()
    shifted = np.zeros_like(gt)
    shifted[4:10, 6:12] = True
    assert dice_score(perfect, gt) == 1.0
    assert iou_score(perfect, gt) == 1.0
    assert boundary_dice_score(perfect, gt, 2) == 1.0
    assert dice_score(shifted, gt) < 1.0
    assert iou_score(shifted, gt) < 1.0
    assert boundary_dice_score(shifted, gt, 2) < 1.0
    assert hd95_score(perfect, gt) == 0.0
    assert assd_score(perfect, gt) == 0.0
    print("eval_boundary_search self-test passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="model checkpoint path")
    parser.add_argument("--output_dir", type=str, help="output directory; must not already exist")
    parser.add_argument("--run_name", type=str, default="boundary_eval")
    parser.add_argument("--data_dir", type=str, default="./data/FetalHeart_v2_high_structure/train_npz")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_FetalHeart_v2_high_structure/")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--n_skip", type=int, default=3)
    parser.add_argument("--vit_patches_size", type=int, default=16)
    parser.add_argument("--model_type", choices=["original", "roi_refiner",
        "baseline_unet", "baseline_attunet", "baseline_unetpp", "baseline_swinunet"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary_weight", type=str, default="original")
    parser.add_argument("--boundary_radius", type=str, default="original")
    parser.add_argument("--boundary_metric_radius", type=int, default=2)
    parser.add_argument("--expect_macro_dice", type=float, default=None)
    parser.add_argument("--expect_macro_iou", type=float, default=None)
    parser.add_argument("--expect_tolerance", type=float, default=1e-6)
    parser.add_argument("--save_labelmaps", action="store_true", default=True)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.checkpoint or not args.output_dir:
        parser.error("--checkpoint and --output_dir are required unless --self_test is used")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        evaluate(parsed)
