"""Aggregate five-seed baseline results in the same format as the AISH tables.

Reads
  baselines_results/test/baseline_<arch>_384_ep70_seed<seed>.json
  baselines_results/boundary/baseline_<arch>_384_ep70_seed<seed>/case_metrics.csv
and the existing Adjusted TransUNet / ROI per-seed values (for paired
comparisons) from summary_5seeds/per_seed_metrics.csv when available.

Outputs to baselines_results/summary_5seeds/:
  per_seed_metrics.csv, five_seed_paper_table.csv, paired_vs_adjusted.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 456, 789, 1024)
METRICS = ("dice", "iou", "recall", "precision", "boundary_dice", "hd95", "assd", "detection_rate")
ARCHS = (
    ("U-Net", "unet"),
    ("Attention U-Net", "attunet"),
    ("UNet++", "unetpp"),
    ("Swin-Unet", "swinunet"),
)
RUN_TEMPLATE = "baseline_{arch}_384_ep70_seed{seed}"


def load_baseline_seed(arch: str, seed: int) -> dict[str, float]:
    run = RUN_TEMPLATE.format(arch=arch, seed=seed)
    regional_path = ROOT / "baselines_results" / "test" / f"{run}.json"
    boundary_dir = ROOT / "baselines_results" / "boundary" / run
    if not regional_path.is_file():
        raise FileNotFoundError(regional_path)
    if not (boundary_dir / "case_metrics.csv").is_file():
        raise FileNotFoundError(boundary_dir / "case_metrics.csv")
    regional = json.loads(regional_path.read_text(encoding="utf-8"))
    values = {
        "dice": float(regional["mean_dice"]),
        "iou": float(regional["mean_iou"]),
        "recall": float(regional["mean_recall"]),
        "precision": float(regional["mean_precision"]),
    }
    values.update(boundary_metrics(boundary_dir))
    return values


def boundary_metrics(folder: Path) -> dict[str, float]:
    with (folder / "case_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["gt_pixels"]) > 0]
    detected = [row for row in rows if int(row["pred_pixels"]) > 0]
    if not rows:
        raise RuntimeError(f"no GT-present rows in {folder}")
    return {
        "boundary_dice": statistics.mean(float(row["boundary_dice"]) for row in rows),
        "hd95": statistics.mean(float(row["hd95"]) for row in detected if row["hd95"]),
        "assd": statistics.mean(float(row["assd"]) for row in detected if row["assd"]),
        "detection_rate": len(detected) / len(rows),
    }


def load_reference_per_seed(label: str) -> dict[int, dict[str, float]] | None:
    """Adjusted/ROI per-seed values from the existing five-seed summary."""
    path = ROOT / "summary_5seeds" / "per_seed_metrics.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == label]
    if not rows:
        return None
    return {int(row["seed"]): {m: float(row[m]) for m in METRICS} for row in rows}


def interval(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def scaled(metric: str, value: float) -> float:
    return value if metric in {"hd95", "assd"} else value * 100.0


def main() -> None:
    output_dir = ROOT / "baselines_results" / "summary_5seeds"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_values: dict[str, dict[int, dict[str, float]]] = {}
    long_rows = []
    compact_rows = []

    for label, arch in ARCHS:
        per_seed = {seed: load_baseline_seed(arch, seed) for seed in SEEDS}
        all_values[label] = per_seed
        for seed, values in per_seed.items():
            long_rows.append({"stage": label, "seed": seed, **values})
        compact = {"stage": label, "n_seeds": len(SEEDS)}
        for metric in METRICS:
            result = interval([per_seed[seed][metric] for seed in SEEDS])
            compact[metric] = (
                f"{scaled(metric, result['mean']):.2f} ± {scaled(metric, result['sd']):.2f} "
                f"[{scaled(metric, result['ci_low']):.2f}, {scaled(metric, result['ci_high']):.2f}]"
            )
        compact_rows.append(compact)

    paired_rows = []
    adjusted = load_reference_per_seed("Adjusted TransUNet")
    roi = load_reference_per_seed("ROI Refiner")
    if adjusted is not None:
        comparisons = [("Adjusted TransUNet", adjusted)]
        if roi is not None:
            comparisons.append(("ROI Refiner", roi))
        for label, baseline in comparisons:
            for model_label, _ in ARCHS:
                model_values = all_values[model_label]
                for metric in METRICS:
                    differences = [model_values[seed][metric] - baseline[seed][metric] for seed in SEEDS]
                    result = interval(differences)
                    test = stats.ttest_1samp(differences, popmean=0.0)
                    paired_rows.append({
                        "comparison": f"{model_label} - {label}",
                        "metric": metric,
                        "n_paired_seeds": len(SEEDS),
                        "mean_difference": scaled(metric, result["mean"]),
                        "sd_difference": scaled(metric, result["sd"]),
                        "ci_low": scaled(metric, result["ci_low"]),
                        "ci_high": scaled(metric, result["ci_high"]),
                        "t_statistic": float(test.statistic),
                        "p_value_two_sided": float(test.pvalue),
                    })
    else:
        print("summary_5seeds/per_seed_metrics.csv not found; skip paired comparisons", file=sys.stderr)

    def write_csv(name: str, rows: list[dict]) -> None:
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("per_seed_metrics.csv", long_rows)
    write_csv("five_seed_paper_table.csv", compact_rows)
    if paired_rows:
        write_csv("paired_vs_adjusted.csv", paired_rows)
    metadata = {
        "seeds": list(SEEDS),
        "archs": {label: arch for label, arch in ARCHS},
        "protocol": "384x384, 70 epochs, batch 4, SGD lr0.01 poly0.9 (Swin-Unet: AdamW lr5e-4 wd0.05 with 5-epoch linear warmup then poly0.9, per its original recipe), corrected foreground Dice + CE, legacy augmentation",
        "confidence_interval": "two-sided 95% Student-t interval across independent training seeds",
        "sd": "sample standard deviation across seeds (ddof=1)",
        "paired_test": "two-sided one-sample t test of paired per-seed differences vs Adjusted TransUNet",
        "important_limit": (
            "Seed-level intervals quantify optimization variability; they do not replace a patient/case-level "
            "confidence interval or external validation."
        ),
    }
    (output_dir / "statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"BASELINES_5SEED_SUMMARY_COMPLETE output={output_dir}")


if __name__ == "__main__":
    main()
