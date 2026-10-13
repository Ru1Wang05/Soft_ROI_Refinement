"""Create case-to-view metadata from the fixed split records without repartitioning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


VIEW_NAMES = ["4CHV", "3VTV", "LVOT", "RVOT"]
VIEW_TO_ID = {name: index for index, name in enumerate(VIEW_NAMES)}


def case_name_from_image_id(image_id: str) -> str:
    path = Path(image_id.replace("\\", "/"))
    return "_".join((*path.parts[:-1], path.stem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split_dir = Path(args.split_dir)
    cases = {}
    counts = {split: Counter() for split in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        with open(split_dir / f"{split}_records.json", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            case_name = case_name_from_image_id(record["image_id"])
            raw_view = record.get("view_label", "")
            view_id = VIEW_TO_ID.get(raw_view, -1)
            entry = {"split": split, "view_name": raw_view, "view_id": view_id}
            previous = cases.get(case_name)
            if previous is not None and previous != entry:
                raise ValueError(f"Conflicting metadata for {case_name}: {previous} vs {entry}")
            cases[case_name] = entry
        for case_name, entry in cases.items():
            if entry["split"] == split:
                counts[split][entry["view_name"]] += 1

    payload = {
        "view_names": VIEW_NAMES,
        "unknown_view_id": -1,
        "cases": cases,
        "counts_by_split": {split: dict(count) for split, count in counts.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload["counts_by_split"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
