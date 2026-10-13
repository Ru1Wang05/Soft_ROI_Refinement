"""Aggregate frozen experiment results into the paper tables.

Targets (select one, or `all`):
  main       five-seed main progression table, 95% t CIs, paired ROI-vs-adjusted test
             -> seed_extension_results/summary_5seeds/
  baselines  five-seed CNN/Swin baseline table, paired tests vs Adjusted/ROI
             -> baselines_results/summary_5seeds/
  final3     frozen three-seed final-test table (legacy run names)
             -> final_test_results/summary/

Usage: python summarize.py [main|baselines|final3|all]
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


# ---------- shared helpers ----------

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


def interval(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def scaled(metric: str, value: float) -> float:
    return value if metric in {"hd95", "assd"} else value * 100.0


def write_csv(directory: Path, name: str, rows: list[dict]) -> None:
    with (directory / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(label_a: str, values_a: dict[int, dict], label_b: str, values_b: dict[int, dict]) -> list[dict]:
    rows = []
    for metric in METRICS:
        differences = [values_a[seed][metric] - values_b[seed][metric] for seed in SEEDS]
        result = interval(differences)
        test = stats.ttest_1samp(differences, popmean=0.0)
        rows.append({
            "comparison": f"{label_a} - {label_b}",
            "metric": metric,
            "n_paired_seeds": len(SEEDS),
            "mean_difference": scaled(metric, result["mean"]),
            "sd_difference": scaled(metric, result["sd"]),
            "ci_low": scaled(metric, result["ci_low"]),
            "ci_high": scaled(metric, result["ci_high"]),
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
        })
    return rows


STATISTICS_METADATA = {
    "seeds": list(SEEDS),
    "confidence_interval": "two-sided 95% Student-t interval across independent training seeds",
    "sd": "sample standard deviation across seeds (ddof=1)",
    "important_limit": (
        "Seed-level intervals quantify optimization variability; they do not replace a patient/case-level "
        "confidence interval or external validation."
    ),
}


# ---------- main five-seed progression table ----------

MAIN_STAGES = (
    ("Official raw", "raw", "official_raw_224_legacy_nopretrain"),
    ("Corrected Dice", "factor", "factor_corrected_nopretrain_224"),
    ("ImageNet-21K", "factor", "factor_legacy_pretrain_224"),
    ("Adjusted TransUNet", "final", "original"),
    ("ROI Refiner", "final", "roi_refiner"),
)
NEW_SEEDS = {789, 1024}


def _main_stage_paths(kind: str, base: str, seed: int) -> tuple[Path, Path]:
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
        test_candidates = (
            ROOT / "factor_test_results" / f"{base}_seed{seed}.json",
            ROOT / "optimization_runs" / "factor_test_results" / f"{base}_seed{seed}.json",
        )
        boundary_candidates = (
            ROOT / "factor_boundary_results" / f"{base}_seed{seed}",
            ROOT / "optimization_runs" / "factor_boundary_results" / f"{base}_seed{seed}",
        )
        return (
            next((path for path in test_candidates if path.is_file()), test_candidates[0]),
            next((path for path in boundary_candidates if path.is_dir()), boundary_candidates[0]),
        )
    return (
        ROOT / "final_test_results" / f"{base}_seed{seed}.json",
        ROOT / "final_boundary_results" / f"{base}_seed{seed}",
    )


def _load_seed(test_path: Path, boundary_dir: Path) -> dict[str, float]:
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    if not (boundary_dir / "case_metrics.csv").is_file():
        raise FileNotFoundError(boundary_dir / "case_metrics.csv")
    regional = json.loads(test_path.read_text(encoding="utf-8"))
    values = {
        "dice": float(regional["mean_dice"]),
        "iou": float(regional["mean_iou"]),
        "recall": float(regional["mean_recall"]),
        "precision": float(regional["mean_precision"]),
    }
    values.update(boundary_metrics(boundary_dir))
    return values


def summarize_main() -> None:
    output_dir = ROOT / "seed_extension_results" / "summary_5seeds"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_values: dict[str, dict[int, dict[str, float]]] = {}
    long_rows = []
    summary_rows = []
    compact_rows = []

    for label, kind, base in MAIN_STAGES:
        per_seed = {seed: _load_seed(*_main_stage_paths(kind, base, seed)) for seed in SEEDS}
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

    write_csv(output_dir, "per_seed_metrics.csv", long_rows)
    write_csv(output_dir, "five_seed_statistics_long.csv", summary_rows)
    write_csv(output_dir, "five_seed_paper_table.csv", compact_rows)
    write_csv(output_dir, "paired_roi_vs_adjusted.csv",
              paired_rows("ROI Refiner", all_values["ROI Refiner"],
                          "Adjusted TransUNet", all_values["Adjusted TransUNet"]))
    metadata = {**STATISTICS_METADATA,
                "paired_test": "two-sided one-sample t test of paired per-seed ROI-minus-adjusted differences"}
    (output_dir / "statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"AISH_5SEED_SUMMARY_COMPLETE output={output_dir}")


# ---------- five-seed baseline table ----------

BASELINE_ARCHS = (
    ("U-Net", "unet"),
    ("Attention U-Net", "attunet"),
    ("UNet++", "unetpp"),
    ("Swin-Unet", "swinunet"),
)


def summarize_baselines() -> None:
    output_dir = ROOT / "baselines_results" / "summary_5seeds"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_values: dict[str, dict[int, dict[str, float]]] = {}
    long_rows = []
    compact_rows = []

    for label, arch in BASELINE_ARCHS:
        run = f"baseline_{arch}_384_ep70_seed{{seed}}"
        per_seed = {}
        for seed in SEEDS:
            test_path = ROOT / "baselines_results" / "test" / run.format(seed=seed)
            boundary_dir = ROOT / "baselines_results" / "boundary" / run.format(seed=seed)
            per_seed[seed] = _load_seed(test_path, boundary_dir)
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

    write_csv(output_dir, "per_seed_metrics.csv", long_rows)
    write_csv(output_dir, "five_seed_paper_table.csv", compact_rows)

    # Paired comparisons against Adjusted TransUNet / ROI Refiner when the
    # main five-seed summary exists.
    reference_path = ROOT / "seed_extension_results" / "summary_5seeds" / "per_seed_metrics.csv"
    if reference_path.is_file():
        with reference_path.open(encoding="utf-8-sig", newline="") as handle:
            reference = [row for row in csv.DictReader(handle)]
        paired = []
        for ref_label in ("Adjusted TransUNet", "ROI Refiner"):
            ref_rows = [row for row in reference if row["stage"] == ref_label]
            ref_values = {int(row["seed"]): {m: float(row[m]) for m in METRICS} for row in ref_rows}
            for label, _ in BASELINE_ARCHS:
                paired.extend(paired_rows(label, all_values[label], ref_label, ref_values))
        if paired:
            write_csv(output_dir, "paired_vs_adjusted.csv", paired)
    else:
        print("main five-seed summary not found; skip paired comparisons", file=sys.stderr)

    metadata = {**STATISTICS_METADATA,
                "archs": {label: arch for label, arch in BASELINE_ARCHS},
                "protocol": "384x384, 70 epochs, batch 4, SGD lr0.01 poly0.9 (Swin-Unet: AdamW lr5e-4 wd0.05 "
                            "with 5-epoch linear warmup then poly0.9, per its original recipe), corrected "
                            "foreground Dice + CE, legacy augmentation",
                "paired_test": "two-sided one-sample t test of paired per-seed differences vs Adjusted TransUNet"}
    (output_dir / "statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BASELINES_5SEED_SUMMARY_COMPLETE output={output_dir}")


# ---------- frozen three-seed final-test table (legacy run names) ----------

FINAL3_EXPECTED = {
    "reference_legacy_224": [42],
    "dicefix_384": [42, 123, 456],
    "dicefix_384_ep70": [42, 123, 456],
}
CLASS_NAMES = {
    1: "Ascending Aorta", 2: "Descending Aorta", 3: "Left Atrium", 4: "Left Ventricle",
    5: "Right Atrium", 6: "Right Ventricle", 7: "Main Pulmonary Artery",
    8: "Left Pulmonary Artery", 9: "Right Pulmonary Artery", 10: "Superior Vena Cava", 11: "Trachea",
}


def summarize_final3(results_root: str = "final_test_results",
                     output_dir: str = "final_test_results/summary") -> None:
    root = ROOT / results_root
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = {method: {} for method in FINAL3_EXPECTED}
    run_rows = []

    for method, seeds in FINAL3_EXPECTED.items():
        for seed in seeds:
            path = root / f"{method}_seed{seed}.json"
            if not path.exists():
                run_rows.append({"method": method, "seed": seed, "status": "missing"})
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("split") != "test" or metrics.get("cases") != 179:
                run_rows.append({"method": method, "seed": seed, "status": "invalid_test_metadata"})
                continue
            data[method][seed] = metrics
            class_dice = [item["mean_dice"] for item in metrics["per_class"].values()]
            run_rows.append({
                "method": method, "seed": seed, "status": "complete",
                "mean_dice": metrics["mean_dice"], "mean_iou": metrics["mean_iou"],
                "mean_recall": metrics["mean_recall"], "mean_precision": metrics["mean_precision"],
                "class_macro_dice": statistics.mean(class_dice),
                "minimum_class_dice": min(class_dice),
            })

    metric_names = ["mean_dice", "mean_iou", "mean_recall", "mean_precision",
                    "class_macro_dice", "minimum_class_dice"]
    summary_rows = []
    class_rows = []
    for method, seeds in FINAL3_EXPECTED.items():
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
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)

        for class_id, class_name in CLASS_NAMES.items():
            values = [run["per_class"][str(class_id)]["mean_dice"] for run in runs]
            if not values:
                continue
            mean, std = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
            class_rows.append({"method": method, "class_id": class_id, "class_name": class_name,
                               "dice_mean": mean, "dice_std": std,
                               "dice_min_run": min(values), "dice_max_run": max(values)})

    write_csv(output, "per_run.csv", run_rows)
    write_csv(output, "summary.csv", summary_rows)
    write_csv(output, "per_class.csv", class_rows)
    print(f"FINAL3_SUMMARY_COMPLETE output={output}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    dispatch = {"main": summarize_main, "baselines": summarize_baselines, "final3": summarize_final3}
    if target == "all":
        targets = ("main", "baselines", "final3")
    elif target in dispatch:
        targets = (target,)
    else:
        sys.exit(f"unknown target: {target} (expected main|baselines|final3|all)")
    for name in targets:
        dispatch[name]()


if __name__ == "__main__":
    main()
