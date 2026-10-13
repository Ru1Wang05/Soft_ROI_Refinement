#!/usr/bin/env python3
"""Create automatically selected qualitative comparison panels from saved predictions."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 123, 456)
TARGET_CLASSES = (8, 9, 10, 11)
CLASS_NAMES = {8: "LPA", 9: "RPA", 10: "SVC", 11: "Trachea"}


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def contour(ax, mask, color, label):
    if mask.any():
        ax.contour(mask.astype(float), levels=[.5], colors=[color], linewidths=1.2)
    ax.plot([], [], color=color, linewidth=1.4, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visual-seed", type=int, default=42, choices=SEEDS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (args.output_dir or root / "paper_results" / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)

    scores = defaultdict(lambda: defaultdict(dict))
    for method in ("original", "roi_refiner"):
        for seed in SEEDS:
            path = root / "final_boundary_results" / f"{method}_seed{seed}" / "case_metrics.csv"
            for row in rows(path):
                cid = int(row["class_id"])
                if cid in TARGET_CLASSES and int(row["gt_pixels"]) > 0:
                    scores[row["case"], cid][method, seed] = float(row["boundary_dice"])

    ranked = []
    for (case, cid), values in scores.items():
        if all((method, seed) in values for method in ("original", "roi_refiner") for seed in SEEDS):
            delta = np.mean([values["roi_refiner", seed] - values["original", seed] for seed in SEEDS])
            ranked.append((float(delta), case, cid))
    if len(ranked) < 4:
        raise RuntimeError("Not enough paired target-class cases for visualization")
    ranked.sort(reverse=True)
    selected = ranked[:3] + [ranked[0 if ranked[-1][0] >= 0 else -1]]

    fig, axes = plt.subplots(len(selected), 4, figsize=(13.5, 3.25 * len(selected)))
    for row_idx, (mean_delta, case, cid) in enumerate(selected):
        data = np.load(args.data_dir / f"{case}.npz")
        image = data["image"]
        gt = data["label"]
        original = np.load(root / "final_boundary_results" / f"original_seed{args.visual_seed}" /
                           "labelmaps" / f"{case}_pred.npy")
        roi = np.load(root / "final_boundary_results" / f"roi_refiner_seed{args.visual_seed}" /
                      "labelmaps" / f"{case}_pred.npy")
        panels = [("Ultrasound + GT", None), ("TransUNet", original),
                  ("ROI Refiner", roi), ("ROI vs. baseline", roi)]
        for col, (title, pred) in enumerate(panels):
            ax = axes[row_idx, col]
            ax.imshow(image, cmap="gray")
            contour(ax, gt == cid, "#00E5FF", "GT")
            if col == 1:
                contour(ax, original == cid, "#FF3B30", "Prediction")
            elif col == 2:
                contour(ax, roi == cid, "#32D74B", "Prediction")
            elif col == 3:
                contour(ax, original == cid, "#FF3B30", "TransUNet")
                contour(ax, roi == cid, "#32D74B", "ROI")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(
            f"{case}\n{CLASS_NAMES[cid]}\nmean ΔBD={mean_delta*100:+.1f} pp", fontsize=9
        )
        axes[row_idx, 3].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Representative target-structure results (3 positive, 1 failure case)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "fig_qualitative_cases.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_qualitative_cases.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out / "qualitative_case_selection.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "case", "class_id", "class_name", "mean_delta_boundary_dice"])
        for i, (delta, case, cid) in enumerate(selected):
            writer.writerow(["positive" if i < 3 else "failure", case, cid, CLASS_NAMES[cid], delta])
    print(f"Wrote qualitative figure to: {out}")


if __name__ == "__main__":
    main()
