#!/usr/bin/env python3
"""
json_prep.py

Stage 5 of the pipeline: consolidate all generated artifacts into a single
dataset manifest.

For each sample, the manifest row includes:
  - caption / prompt text (from theme_prompts + theme_images metadata)
  - ground-truth scene image (theme_images)
  - canvas composite (canvas)
  - depth map + raw depth array (depth)
  - bounding-box annotations (bbox_annotations)
  - per-object reference images (reference_images)

Outputs:
  - outputs/dataset_manifest.jsonl   (default, one JSON object per line)
  - outputs/dataset_manifest.json    (optional, single JSON array)

Typical usage:

    python json_prep.py

    # Only samples with every artifact present:
    python json_prep.py --require_complete

    # Single sample:
    python json_prep.py --theme beach_day_007022

    # Write a single JSON array instead of JSONL:
    python json_prep.py --format json --limit 100

    # Use more parallel workers:
    python json_prep.py --threads 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_THEME_PROMPTS_DIR = DEFAULT_OUTPUTS_DIR / "theme_prompts/samples"
DEFAULT_THEME_IMAGES_DIR = DEFAULT_OUTPUTS_DIR / "theme_images"
DEFAULT_BBOX_DIR = DEFAULT_OUTPUTS_DIR / "bbox_annotations/metadata"
DEFAULT_REFERENCE_DIR = DEFAULT_OUTPUTS_DIR / "reference_images"
DEFAULT_CANVAS_DIR = DEFAULT_OUTPUTS_DIR / "canvas"
DEFAULT_DEPTH_DIR = DEFAULT_OUTPUTS_DIR / "depth"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUTS_DIR / "dataset_manifest.jsonl"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUTS_DIR / "dataset_manifest.json"
DEFAULT_NUM_THREADS = min(16, os.cpu_count() or 8)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a consolidated dataset manifest from pipeline outputs."
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Root outputs directory produced by the v2 pipeline.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Defaults to dataset_manifest.jsonl or .json based on --format.",
    )
    p.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="Output format: jsonl (one sample per line) or json (single array).",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Process only this sample id (e.g. beach_day_007022).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many samples (after filtering).",
    )
    p.add_argument(
        "--require_complete",
        action="store_true",
        help="Only include samples that have gt image, caption, depth, canvas, bbox, and references.",
    )
    p.add_argument(
        "--relative_paths",
        action="store_true",
        help="Store file paths relative to --outputs_dir instead of absolute paths.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"Number of parallel worker threads (default: {DEFAULT_NUM_THREADS}).",
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Path | str | None, outputs_dir: Path, relative_paths: bool) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if relative_paths:
        try:
            return str(resolved.relative_to(outputs_dir.resolve()))
        except ValueError:
            return str(resolved)
    return str(resolved)


def collect_reference_images(
    sample_id: str,
    bbox_meta: dict | None,
    reference_dir: Path,
    outputs_dir: Path,
    relative_paths: bool,
) -> list[dict]:
    sample_ref_dir = reference_dir / sample_id
    if not sample_ref_dir.is_dir():
        return []

    bbox_objects = []
    if bbox_meta:
        bbox_objects = bbox_meta.get("bbox_information", {}).get("objects", [])

    references: list[dict] = []
    for ref_meta_path in sorted(sample_ref_dir.glob("bbox*.json")):
        ref_meta = load_json(ref_meta_path)
        bbox_index = ref_meta.get("bbox_index")
        if bbox_index is None:
            stem = ref_meta_path.stem
            if stem.startswith("bbox") and stem[4:].isdigit():
                bbox_index = int(stem[4:])

        bbox_entry = None
        if bbox_index is not None and 1 <= bbox_index <= len(bbox_objects):
            bbox_entry = bbox_objects[bbox_index - 1]

        ref_image_path = ref_meta.get("output_image") or str(
            sample_ref_dir / f"bbox{bbox_index}.jpg"
        )

        references.append(
            {
                "bbox_index": bbox_index,
                "object": ref_meta.get("object") or (bbox_entry or {}).get("object"),
                "detected_label": ref_meta.get("detected_label")
                or (bbox_entry or {}).get("label"),
                "detection_score": ref_meta.get("detection_score")
                or (bbox_entry or {}).get("score"),
                "source_bbox": ref_meta.get("source_bbox") or (bbox_entry or {}).get("bbox"),
                "source_bbox_norm": (bbox_entry or {}).get("bbox_norm"),
                "image": resolve_path(ref_image_path, outputs_dir, relative_paths),
                "metadata": resolve_path(ref_meta_path, outputs_dir, relative_paths),
                "seed": ref_meta.get("seed"),
                "model_id": ref_meta.get("model_id"),
            }
        )

    references.sort(key=lambda item: item.get("bbox_index") or 0)
    return references


def build_sample_record(
    sample_id: str,
    paths: dict[str, Path],
    outputs_dir: Path,
    relative_paths: bool,
    require_complete: bool,
) -> dict | None:
    theme_prompt_path = paths["theme_prompts"] / f"{sample_id}.json"
    theme_meta_path = paths["theme_images_meta"] / f"{sample_id}.json"
    gt_image_path = paths["theme_images"] / f"{sample_id}.png"
    bbox_meta_path = paths["bbox"] / f"{sample_id}.json"
    canvas_meta_path = paths["canvas_meta"] / f"{sample_id}.json"
    canvas_image_path = paths["canvas_images"] / f"{sample_id}.png"
    depth_meta_path = paths["depth_meta"] / f"{sample_id}.json"
    depth_image_path = paths["depth_images"] / f"{sample_id}.png"
    depth_raw_path = paths["depth_raw"] / f"{sample_id}.npy"

    required_paths = {
        "gt_image": gt_image_path,
        "theme_metadata": theme_meta_path,
        "bbox_metadata": bbox_meta_path,
        "depth_image": depth_image_path,
    }
    if require_complete:
        required_paths.update(
            {
                "theme_prompt": theme_prompt_path,
                "canvas_image": canvas_image_path,
                "canvas_metadata": canvas_meta_path,
                "depth_metadata": depth_meta_path,
                "depth_raw": depth_raw_path,
            }
        )

    for label, path in required_paths.items():
        if not path.is_file():
            return None

    theme_meta = load_json(theme_meta_path) if theme_meta_path.is_file() else {}
    theme_prompt = load_json(theme_prompt_path) if theme_prompt_path.is_file() else {}
    bbox_meta = load_json(bbox_meta_path) if bbox_meta_path.is_file() else None
    canvas_meta = load_json(canvas_meta_path) if canvas_meta_path.is_file() else None
    depth_meta = load_json(depth_meta_path) if depth_meta_path.is_file() else None

    references = collect_reference_images(
        sample_id=sample_id,
        bbox_meta=bbox_meta,
        reference_dir=paths["reference"],
        outputs_dir=outputs_dir,
        relative_paths=relative_paths,
    )
    if require_complete and not references:
        return None

    caption = (
        theme_meta.get("prompt")
        or theme_prompt.get("combined_prompt")
        or theme_prompt.get("prompt")
    )
    structured_prompt = theme_meta.get("structured_prompt") or theme_prompt.get(
        "structured_prompt"
    )

    record = {
        "id": sample_id,
        "theme": theme_meta.get("theme") or theme_prompt.get("theme") or sample_id.rsplit("_", 1)[0],
        "caption": caption,
        "structured_prompt": structured_prompt,
        "num_objects": theme_meta.get("num_objects") or theme_prompt.get("num_objects"),
        "objects": theme_meta.get("objects") or theme_prompt.get("objects") or [],
        "object_details": theme_meta.get("object_details")
        or theme_prompt.get("object_details")
        or [],
        "gt_image": resolve_path(gt_image_path, outputs_dir, relative_paths),
        "canvas_image": resolve_path(canvas_image_path, outputs_dir, relative_paths)
        if canvas_image_path.is_file()
        else None,
        "depth_image": resolve_path(depth_image_path, outputs_dir, relative_paths)
        if depth_image_path.is_file()
        else None,
        "depth_raw": resolve_path(depth_raw_path, outputs_dir, relative_paths)
        if depth_raw_path.is_file()
        else None,
        "image_width": theme_meta.get("gen_width")
        or (bbox_meta or {}).get("image_width")
        or (depth_meta or {}).get("image_width"),
        "image_height": theme_meta.get("gen_height")
        or (bbox_meta or {}).get("image_height")
        or (depth_meta or {}).get("image_height"),
        "generation": {
            "seed": theme_meta.get("seed"),
            "guidance_scale": theme_meta.get("guidance_scale"),
            "num_inference_steps": theme_meta.get("num_inference_steps"),
            "model_id": theme_meta.get("model_id"),
        },
        "bbox_annotations": (bbox_meta or {}).get("bbox_information"),
        "canvas": {
            "image": resolve_path(canvas_image_path, outputs_dir, relative_paths)
            if canvas_image_path.is_file()
            else None,
            "metadata": resolve_path(canvas_meta_path, outputs_dir, relative_paths)
            if canvas_meta_path.is_file()
            else None,
            "num_placements": (canvas_meta or {}).get("num_placements"),
            "placements": (canvas_meta or {}).get("placements"),
        },
        "depth": {
            "image": resolve_path(depth_image_path, outputs_dir, relative_paths)
            if depth_image_path.is_file()
            else None,
            "raw": resolve_path(depth_raw_path, outputs_dir, relative_paths)
            if depth_raw_path.is_file()
            else None,
            "metadata": resolve_path(depth_meta_path, outputs_dir, relative_paths)
            if depth_meta_path.is_file()
            else None,
            "depth_type": (depth_meta or {}).get("depth_type"),
            "model_id": (depth_meta or {}).get("model_id"),
            "depth_min": (depth_meta or {}).get("depth_min"),
            "depth_max": (depth_meta or {}).get("depth_max"),
        },
        "reference_images": references,
        "source_files": {
            "theme_prompt": resolve_path(theme_prompt_path, outputs_dir, relative_paths)
            if theme_prompt_path.is_file()
            else None,
            "theme_metadata": resolve_path(theme_meta_path, outputs_dir, relative_paths)
            if theme_meta_path.is_file()
            else None,
            "bbox_metadata": resolve_path(bbox_meta_path, outputs_dir, relative_paths)
            if bbox_meta_path.is_file()
            else None,
        },
    }

    return record


def process_sample_safe(
    sample_id: str,
    paths: dict[str, Path],
    outputs_dir: Path,
    relative_paths: bool,
    require_complete: bool,
) -> tuple[str, dict | None, str | None]:
    try:
        record = build_sample_record(
            sample_id=sample_id,
            paths=paths,
            outputs_dir=outputs_dir,
            relative_paths=relative_paths,
            require_complete=require_complete,
        )
        if record is None:
            return sample_id, None, "missing required artifacts"
        return sample_id, record, None
    except Exception as e:
        return sample_id, None, f"ERROR: {e}"


def discover_sample_ids(paths: dict[str, Path], theme: str | None) -> list[str]:
    if theme:
        return [theme]

    sample_ids = {
        path.stem
        for path in paths["theme_images_meta"].glob("*.json")
        if path.is_file()
    }
    return sorted(sample_ids)


def write_manifest(records: list[dict], output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    output_path = Path(
        args.output
        or (DEFAULT_OUTPUT_JSONL if args.format == "jsonl" else DEFAULT_OUTPUT_JSON)
    ).resolve()

    paths = {
        "theme_prompts": outputs_dir / "theme_prompts/samples",
        "theme_images": outputs_dir / "theme_images/images",
        "theme_images_meta": outputs_dir / "theme_images/metadata",
        "bbox": outputs_dir / "bbox_annotations/metadata",
        "reference": outputs_dir / "reference_images",
        "canvas_images": outputs_dir / "canvas/images",
        "canvas_meta": outputs_dir / "canvas/metadata",
        "depth_images": outputs_dir / "depth/images",
        "depth_meta": outputs_dir / "depth/metadata",
        "depth_raw": outputs_dir / "depth/raw",
    }

    sample_ids = discover_sample_ids(paths, args.theme)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    if not sample_ids:
        print("ERROR: No samples found to process.", file=sys.stderr)
        return 1

    print(f"Samples to process: {len(sample_ids)} (threads={args.threads})")

    records_by_id: dict[str, dict] = {}
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                process_sample_safe,
                sample_id,
                paths,
                outputs_dir,
                args.relative_paths,
                args.require_complete,
            ): sample_id
            for sample_id in sample_ids
        }
        with tqdm(total=len(sample_ids), desc="Building dataset manifest", unit="sample") as pbar:
            for future in as_completed(futures):
                sample_id, record, detail = future.result()
                if record is not None:
                    records_by_id[sample_id] = record
                elif detail and detail.startswith("ERROR:"):
                    failed += 1
                    tqdm.write(f"[{sample_id}] {detail}", file=sys.stderr)
                else:
                    skipped += 1

                pbar.update(1)
                pbar.set_postfix(wrote=len(records_by_id), skip=skipped, fail=failed, refresh=False)

    records = [records_by_id[sample_id] for sample_id in sample_ids if sample_id in records_by_id]

    if not records:
        print("ERROR: No samples matched the requested filters.", file=sys.stderr)
        return 1

    write_manifest(records, output_path, args.format)

    print(
        f"Wrote {len(records)} samples to {output_path} "
        f"(skipped={skipped}, failed={failed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
