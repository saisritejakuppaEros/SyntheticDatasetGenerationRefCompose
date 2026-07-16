#!/usr/bin/env python3
"""
Offline helper: run an external layout tool to produce bbox CSVs for CanvasSceneDataset.

Example:
  python prepare_dataset.py \\
    --layout_command "python /path/to/layout_zip/generate_layout.py --scenes_dir ./scenes --out ./layout_out" \\
    --merge_csv ./detections.csv \\
    --output_csv ./dataset_with_layout.csv

If you already have a CSV with image_path, x1, y1, x2, y2, you can skip this script and pass it
directly to training via --dataset_type canvas --csv_path ...
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Prepare canvas dataset CSV (layout pre-pass wrapper).")
    p.add_argument(
        "--layout_command",
        type=str,
        default="",
        help="Shell command to run your layout zip (e.g. generate_layout.py). Leave empty to only merge CSVs.",
    )
    p.add_argument(
        "--merge_csv",
        type=str,
        default="",
        help="Optional existing CSV to merge with layout output (matched on image_path).",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path for training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.layout_command:
        ret = subprocess.run(args.layout_command, shell=True)
        if ret.returncode != 0:
            print("layout_command failed", file=sys.stderr)
            sys.exit(ret.returncode)

    if args.merge_csv:
        merge = Path(args.merge_csv)
        if not merge.is_file():
            raise FileNotFoundError(merge)
        rows = []
        with open(merge, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path} ({len(rows)} rows) from merge only.")
    elif not args.layout_command:
        print("Nothing to do: set --layout_command and/or --merge_csv.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Layout command finished. Copy or post-process its output into {out_path} as needed.")


if __name__ == "__main__":
    main()
