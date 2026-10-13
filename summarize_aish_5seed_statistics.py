"""Aggregate five frozen seeds, 95% t CIs, and paired ROI-vs-baseline effects."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 456, 789, 1024)
NEW_SEEDS = {789, 1024}
METRICS = (
    "dice", "iou", "recall", "precision", "boundary_dice", "hd95", "assd", "detection_rate"
)
STAGES = (
    ("Official raw", "raw", "official_raw_224_legacy_nopretrain"),
    ("Corrected Dice", "factor", "factor_corrected_nopretrain_224"),
    ("ImageNet-21K", "factor", "factor_legacy_pretrain_224"),
    ("Adjusted TransUNet", "final", "original"),
    ("ROI Refiner", "final", "roi_refiner"),
)


def existing_paths(kind: str, base: str, seed: int) -> tuple[Path, Path]:
    if seed in NEW_SEEDS:
        # The final adjusted baseline is stored under its training run name
        # (dicefix_384_ep70), while legacy three-seed test files use
        # original_seed*.json. ROI keeps its own run name.
        if kind == "final" and base == "original":
            run_base = "dicefix_384_ep70"
        elif kind == "final" and base == "roi_refiner":
            run_base = "roi_refiner_384_ep70"
        else:
            run_base = base
        run = f"{run_base}_seed{seed}"
        return (
            ROOT / "seed_extension_results" / "test" / f"{run}.json",
            ROOT / "seed_extension_results" / "boundary" / run,
        )
    if kind == "raw":
        return (
            ROOT / "raw_test_results" / f"{base}_seed{seed}.json",
            ROOT / "raw_boundary_results" / f"{base}_seed{seed}",
        )
    if kind == "factor":
        # Older factorial scripts write these folders at project root; some
        # transfers place them below optimization_runs. Accept both layouts.
        test_name = f"{base}_seed{seed}.json"
        boundary_name = f"{base}_seed{seed}"
        test_candidates = (
            ROOT / "factor_test_results" / test_name,
            ROOT / "optimization_runs" / "factor_test_results" / test_name,
        )
        boundary_candidates = (
            ROOT / "factor_boundary_results" / boundary_name,
            ROOT / "optimization_runs" / "factor_boundary_results" / boundary_name,
        )
        test_path = next((path for path in test_candidates if path.is_file()), test_candidates[0])
        boundary_path = next((path for path in boundary_candidates if path.is_dir()), boundary_candidates[0])
        return test_path, boundary_path
    return (
        ROOT / "final_test_results" / f"{base}_seed{seed}.json",
        ROOT / "final_boundary_results" / f"{base}_seed{seed}",
    )


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


def load_seed(kind: str, base: str, seed: int) -> dict[str, float]:
    regional_path, boundary_dir = existing_paths(kind, base, seed)
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


def interval(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def scaled(metric: str, value: float) -> float:
    return value if metric in {"hd95", "assd"} else value * 100.0


def main() -> None:
    output_dir = ROOT / "seed_extension_results" / "summary_5seeds"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_values: dict[str, dict[int, dict[str, float]]] = {}
    long_rows = []
    summary_rows = []
    compact_rows = []

    for label, kind, base in STAGES:
        per_seed = {seed: load_seed(kind, base, seed) for seed in SEEDS}
        all_values[label] = per_seed
        for seed, values in per_seed.items():
            long_rows.append({"stage": label, "seed": seed, **values})

        compact = {"stage": label, "n_seeds": len(SEEDS)}
        for metric in METRICS:
            result = interval([per_seed[seed][metric] for seed in SEEDS])
            row = {"stage": label, "metric": metric}
            row.update({key: scaled(metric, value) if key != "n" else value for key, value in result.items()})
            summary_rows.append(row)
            compact[metric] = (
                f"{scaled(metric, result['mean']):.2f} ± {scaled(metric, result['sd']):.2f} "
                f"[{scaled(metric, result['ci_low']):.2f}, {scaled(metric, result['ci_high']):.2f}]"
            )
        compact_rows.append(compact)

    paired_rows = []
    baseline = all_values["Adjusted TransUNet"]
    roi = all_values["ROI Refiner"]
    for metric in METRICS:
        differences = [roi[seed][metric] - baseline[seed][metric] for seed in SEEDS]
        result = interval(differences)
        test = stats.ttest_1samp(differences, popmean=0.0)
        paired_rows.append({
            "comparison": "ROI Refiner - Adjusted TransUNet",
            "metric": metric,
            "n_paired_seeds": len(SEEDS),
            "mean_difference": scaled(metric, result["mean"]),
            "sd_difference": scaled(metric, result["sd"]),
            "ci_low": scaled(metric, result["ci_low"]),
            "ci_high": scaled(metric, result["ci_high"]),
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
        })

    def write_csv(name: str, rows: list[dict]) -> None:
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("per_seed_metrics.csv", long_rows)
    write_csv("five_seed_statistics_long.csv", summary_rows)
    write_csv("five_seed_paper_table.csv", compact_rows)
    write_csv("paired_roi_vs_adjusted.csv", paired_rows)
    metadata = {
        "seeds": list(SEEDS),
        "confidence_interval": "two-sided 95% Student-t interval across independent training seeds",
        "sd": "sample standard deviation across seeds (ddof=1)",
        "paired_test": "two-sided one-sample t test of paired per-seed ROI-minus-adjusted differences",
        "important_limit": (
            "Seed-level intervals quantify optimization variability; they do not replace a patient/case-level "
            "confidence interval or external validation."
        ),
    }
    (output_dir / "statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"AISH_5SEED_SUMMARY_COMPLETE output={output_dir}")


if __name__ == "__main__":
    main()
