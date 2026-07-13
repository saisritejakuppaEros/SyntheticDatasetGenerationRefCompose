#!/usr/bin/env python3
"""
bbox_annotation.py

Run Grounding DINO zero-shot object detection on theme images and write
bounding boxes to a separate output folder (does not modify theme metadata).

Typical usage:

    CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python bbox_annotation.py

    # Single theme + debug overlay:
    CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python bbox_annotation.py --theme action --show

    # Visualize existing bbox output (no model run):
    ../.venv/bin/python bbox_visualize.py --theme action --show
"""

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_METADATA_DIR = SCRIPT_DIR / "outputs/theme_images/metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/bbox_annotations/metadata"
DEFAULT_DEBUG_DIR = SCRIPT_DIR / "outputs/bbox_annotations/debug_bboxes"
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"

BBOX_DETAIL_KEYS = {
    "bbox_xyxy",
    "bbox_xyxy_norm",
    "bbox_score",
    "bbox_label",
    "bbox_found",
    "bbox_information",
}
BBOX_TOP_LEVEL_KEYS = {
    "bbox_annotations",
    "bbox_information",
    "bbox_model_id",
    "bbox_threshold",
    "bbox_text_threshold",
    "bbox_image_width",
    "bbox_image_height",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Annotate theme images with Grounding DINO bounding boxes."
    )
    p.add_argument(
        "--metadata_dir",
        type=str,
        default=str(DEFAULT_SOURCE_METADATA_DIR),
        help="Directory containing source theme metadata JSON files (read-only).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where bbox annotation JSON files are written.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id for Grounding DINO.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Box confidence threshold for post-processing.",
    )
    p.add_argument(
        "--text_threshold",
        type=float,
        default=0.3,
        help="Text alignment threshold for post-processing.",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Process only this theme (metadata filename stem).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip themes whose bbox output JSON already exists (default: True).",
    )
    p.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Save a bbox overlay PNG after each theme (see bbox_visualize.py).",
    )
    p.add_argument(
        "--debug_dir",
        type=str,
        default=str(DEFAULT_DEBUG_DIR),
        help="Where to write debug overlay PNGs when --show is set.",
    )
    return p.parse_args()


def indefinite_article(noun: str) -> str:
    return "an" if noun[0].lower() in "aeiou" else "a"


def build_detection_text(objects: list[str]) -> tuple[str, list[list[str]]]:
    phrases = [f"{indefinite_article(obj)} {obj}" for obj in objects]
    text = " ".join(f"{phrase}." for phrase in phrases)
    text_labels = [phrases]
    return text, text_labels


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"^(a|an)\s+", "", label)
    label = re.sub(r"[\s\-]+", " ", label)
    return label.strip()


def labels_match(expected_object: str, detected_label: str) -> bool:
    expected = normalize_label(expected_object)
    detected = normalize_label(detected_label)
    return expected == detected or expected in detected or detected in expected


def box_to_record(box, score, detected_label, image_width, image_height):
    xmin, ymin, xmax, ymax = [float(v) for v in box]
    xyxy = {
        "xmin": round(xmin, 2),
        "ymin": round(ymin, 2),
        "xmax": round(xmax, 2),
        "ymax": round(ymax, 2),
    }
    xyxy_norm = {
        "xmin": round(xmin / image_width, 6),
        "ymin": round(ymin / image_height, 6),
        "xmax": round(xmax / image_width, 6),
        "ymax": round(ymax / image_height, 6),
    }
    return {
        "found": True,
        "label": detected_label,
        "score": round(float(score), 6),
        "bbox": [xyxy["xmin"], xyxy["ymin"], xyxy["xmax"], xyxy["ymax"]],
        "bbox_norm": [
            xyxy_norm["xmin"],
            xyxy_norm["ymin"],
            xyxy_norm["xmax"],
            xyxy_norm["ymax"],
        ],
        "xyxy": xyxy,
        "xyxy_norm": xyxy_norm,
    }


def empty_bbox_record():
    return {
        "found": False,
        "label": None,
        "score": None,
        "bbox": None,
        "bbox_norm": None,
        "xyxy": None,
        "xyxy_norm": None,
    }


def assign_bboxes(objects, detections, image_width, image_height):
    scores = detections.get("scores", [])
    boxes = detections.get("boxes", [])
    labels = detections.get("text_labels") or detections.get("labels") or []

    assigned = []
    for obj in objects:
        best = None
        for idx, label in enumerate(labels):
            label_text = label if isinstance(label, str) else str(label)
            if not labels_match(obj, label_text):
                continue
            score = float(scores[idx])
            if best is None or score > best[0]:
                best = (score, boxes[idx], label_text)

        if best is None:
            assigned.append({"object": obj, **empty_bbox_record()})
        else:
            assigned.append(
                {
                    "object": obj,
                    **box_to_record(best[1], best[0], best[2], image_width, image_height),
                }
            )
    return assigned


def build_bbox_output(
    source_metadata: dict,
    source_meta_path: Path,
    bbox_assignments: list[dict],
    model_id: str,
    threshold: float,
    text_threshold: float,
    image_width: int,
    image_height: int,
) -> dict:
    return {
        "id": source_metadata.get("id"),
        "theme": source_metadata.get("theme"),
        "num_objects": source_metadata.get("num_objects"),
        "objects": source_metadata.get("objects", []),
        "source_metadata": str(source_meta_path.resolve()),
        "source_image": source_metadata.get("output_image"),
        "bbox_model_id": model_id,
        "bbox_threshold": threshold,
        "bbox_text_threshold": text_threshold,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_information": {
            "model_id": model_id,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "image_width": image_width,
            "image_height": image_height,
            "objects": bbox_assignments,
        },
    }


def load_model(model_id: str, device: str):
    print(f"Loading Grounding DINO ({model_id}) on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def detect_objects(processor, model, image, objects, device, threshold, text_threshold):
    text, text_labels = build_detection_text(objects)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
        text_labels=text_labels,
    )
    return results[0]


def annotate_theme(source_meta_path: Path, output_meta_path: Path, processor, model, args):
    with open(source_meta_path, "r", encoding="utf-8") as f:
        source_metadata = json.load(f)

    theme = source_metadata.get("theme", source_meta_path.stem)
    image_path = Path(source_metadata.get("output_image", ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for {theme}: {image_path}")

    objects = source_metadata.get("objects") or []
    if not objects:
        raise ValueError(f"No objects listed in source metadata for {theme}")

    image = Image.open(image_path).convert("RGB")
    detections = detect_objects(
        processor,
        model,
        image,
        objects,
        args.device,
        args.threshold,
        args.text_threshold,
    )
    bbox_assignments = assign_bboxes(objects, detections, image.width, image.height)
    bbox_output = build_bbox_output(
        source_metadata,
        source_meta_path,
        bbox_assignments,
        args.model_id,
        args.threshold,
        args.text_threshold,
        image.width,
        image.height,
    )

    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(bbox_output, f, indent=2, ensure_ascii=False)

    found = sum(1 for item in bbox_assignments if item.get("found"))
    debug_path = None
    if args.show:
        from bbox_visualize import visualize_metadata_file

        debug_path = visualize_metadata_file(
            output_meta_path,
            debug_dir=Path(args.debug_dir),
            show=True,
        )

    return theme, found, len(bbox_assignments), output_meta_path, debug_path


def main():
    args = parse_args()
    source_metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    if not source_metadata_dir.is_dir():
        print(f"ERROR: source metadata dir not found: {source_metadata_dir}", file=sys.stderr)
        sys.exit(1)

    source_paths = sorted(source_metadata_dir.glob("*.json"))
    if args.theme:
        source_paths = [source_metadata_dir / f"{args.theme}.json"]
        if not source_paths[0].is_file():
            print(f"ERROR: source metadata not found for theme: {args.theme}", file=sys.stderr)
            sys.exit(1)

    processor, model = load_model(args.model_id, args.device)

    n_done, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()

    for source_meta_path in source_paths:
        theme = source_meta_path.stem
        output_meta_path = output_dir / f"{theme}.json"

        try:
            if args.skip_existing and output_meta_path.is_file():
                if args.show:
                    from bbox_visualize import visualize_metadata_file

                    debug_path = visualize_metadata_file(
                        output_meta_path,
                        debug_dir=Path(args.debug_dir),
                        show=True,
                    )
                    print(
                        f"[{theme}] SKIP annotate; debug overlay -> {debug_path.name}"
                    )
                else:
                    print(f"[{theme}] SKIP: bbox output already exists -> {output_meta_path}")
                n_skipped += 1
                continue

            theme, found, total, out_path, debug_path = annotate_theme(
                source_meta_path,
                output_meta_path,
                processor,
                model,
                args,
            )
            n_done += 1
            msg = f"[{theme}] annotated {found}/{total} objects -> {out_path}"
            if debug_path is not None:
                msg += f" | debug: {debug_path.name}"
            print(msg)

        except Exception as e:
            n_failed += 1
            print(f"[{theme}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    elapsed = time.time() - t_start
    print(
        f"DONE. annotated={n_done} skipped={n_skipped} failed={n_failed} "
        f"output_dir={output_dir} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
