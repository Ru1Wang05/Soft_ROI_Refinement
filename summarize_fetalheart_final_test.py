"""Summarise the single locked test-set evaluation stage."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


EXPECTED = {
    "reference_legacy_224": [42],
    "dicefix_384": [42, 123, 456],
    "dicefix_384_ep70": [42, 123, 456],
}
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


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="final_test_results")
    parser.add_argument("--output_dir", default="final_test_results/summary")
    args = parser.parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {method: {} for method in EXPECTED}
    run_rows = []

    for method, seeds in EXPECTED.items():
        for seed in seeds:
            path = root / f"{method}_seed{seed}.json"
            if not path.exists():
                run_rows.append({"method": method, "seed": seed, "status": "missing"})
                continue
            with open(path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("split") != "test" or metrics.get("cases") != 179:
                run_rows.append({"method": method, "seed": seed, "status": "invalid_test_metadata"})
                continue
            data[method][seed] = metrics
            class_dice = [item["mean_dice"] for item in metrics["per_class"].values()]
            row = {
                "method": method,
                "seed": seed,
                "status": "complete",
                "mean_dice": metrics["mean_dice"],
                "mean_iou": metrics["mean_iou"],
                "mean_recall": metrics["mean_recall"],
                "mean_precision": metrics["mean_precision"],
                "class_macro_dice": statistics.mean(class_dice),
                "minimum_class_dice": min(class_dice),
            }
            run_rows.append(row)

    metric_names = [
        "mean_dice",
        "mean_iou",
        "mean_recall",
        "mean_precision",
        "class_macro_dice",
        "minimum_class_dice",
    ]
    summary_rows = []
    class_rows = []
    for method, seeds in EXPECTED.items():
        runs = list(data[method].values())
        row = {"method": method, "completed_runs": len(runs), "required_runs": len(seeds)}
        for metric in metric_names:
            if not runs:
                continue
            values = []
            for run in runs:
                if metric == "class_macro_dice":
                    values.append(statistics.mean(item["mean_dice"] for item in run["per_class"].values()))
                elif metric == "minimum_class_dice":
                    values.append(min(item["mean_dice"] for item in run["per_class"].values()))
                else:
                    values.append(run[metric])
            mean, std = mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)

        for class_id, class_name in CLASS_NAMES.items():
            values = [run["per_class"][str(class_id)]["mean_dice"] for run in runs]
            if not values:
                continue
            mean, std = mean_std(values)
            class_rows.append(
                {
                    "method": method,
                    "class_id": class_id,
                    "class_name": class_name,
                    "dice_mean": mean,
                    "dice_std": std,
                    "dice_min_run": min(values),
                    "dice_max_run": max(values),
                }
            )

    run_fields = ["method", "seed", "status"] + metric_names
    with open(output / "per_run.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)
    summary_fields = ["method", "completed_runs", "required_runs"]
    for metric in metric_names:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    with open(output / "summary.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(output / "per_class.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "class_id",
                "class_name",
                "dice_mean",
                "dice_std",
                "dice_min_run",
                "dice_max_run",
            ],
        )
        writer.writeheader()
        writer.writerows(class_rows)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
