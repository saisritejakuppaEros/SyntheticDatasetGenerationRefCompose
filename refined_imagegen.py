#!/usr/bin/env python3
"""Analyze stage3 metadata: flag samples with 2+ landmark references as bad."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parent / "outputs/stage3_generated/metadata"

# Reference paths look like: outputs/<category>/images/...
# Stage 1 stores landscapes under "landmark".
CATEGORIES = ("landmark", "cuisine", "object", "celeb", "person")


def category_from_path(path: str) -> str:
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part == "outputs" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def analyze_metadata(metadata_dir: Path = METADATA_DIR) -> dict:
    total = 0
    good = 0
    bad = 0
    landmark_count_dist = Counter()
    category_combo_dist = Counter()
    bad_samples: list[dict] = []

    for json_path in sorted(metadata_dir.glob("*.json")):
        with json_path.open() as f:
            sample = json.load(f)

        refs = sample.get("reference_images", [])
        categories = [category_from_path(p) for p in refs]
        landmark_count = sum(1 for c in categories if c == "landmark")

        total += 1
        landmark_count_dist[landmark_count] += 1
        category_combo_dist[tuple(sorted(categories))] += 1

        if landmark_count >= 2:
            bad += 1
            if len(bad_samples) < 10:
                bad_samples.append(
                    {
                        "id": sample.get("id", json_path.stem),
                        "landmark_count": landmark_count,
                        "categories": categories,
                        "reference_images": refs,
                    }
                )
        else:
            good += 1

    return {
        "metadata_dir": str(metadata_dir),
        "total": total,
        "good": good,
        "bad": bad,
        "good_pct": 100.0 * good / total if total else 0.0,
        "bad_pct": 100.0 * bad / total if total else 0.0,
        "landmark_count_dist": dict(sorted(landmark_count_dist.items())),
        "category_combo_dist": {
            " + ".join(combo): count
            for combo, count in category_combo_dist.most_common()
        },
        "bad_examples": bad_samples,
    }


def print_report(report: dict) -> None:
    print("=" * 72)
    print("Stage3 metadata quality analysis")
    print("Rule: GOOD = at most 1 landmark ref | BAD = 2+ landmark refs")
    print("=" * 72)
    print(f"Metadata dir : {report['metadata_dir']}")
    print(f"Total samples: {report['total']:,}")
    print(f"Good samples : {report['good']:,} ({report['good_pct']:.2f}%)")
    print(f"Bad samples  : {report['bad']:,} ({report['bad_pct']:.2f}%)")
    print()
    print("Landmark count distribution:")
    for count, n in report["landmark_count_dist"].items():
        label = "good" if count < 2 else "bad"
        print(f"  {count} landmark(s): {n:,} samples ({label})")
    print()
    print("Category combinations (top 15):")
    for combo, n in list(report["category_combo_dist"].items())[:15]:
        print(f"  {combo}: {n:,}")
    if report["bad_examples"]:
        print()
        print("Example bad samples (up to 10):")
        for ex in report["bad_examples"]:
            print(f"  - {ex['id']}: {ex['landmark_count']} landmarks -> {ex['categories']}")


if __name__ == "__main__":
    report = analyze_metadata()
    print_report(report)
