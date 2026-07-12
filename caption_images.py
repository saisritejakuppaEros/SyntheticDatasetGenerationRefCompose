"""
Stage 1: Reference Image Quality Filtering + Captioning Pipeline
------------------------------------------------------------------
For each category (celeb, object, cuisine, landmark):
  1. Reads source images from the configured glob pattern
  2. Sends each image to the local Qwen3-VL server for QUALITY / REASON / CAPTION
  3. Saves a COPY of the image + a paired .json metadata file, preserving the
     source folder hierarchy under outputs/<category>/images/ and /metadata/
  4. Appends every record to a single outputs/manifest.jsonl so the whole
     dataset can be inspected/loaded from one place

Output layout:

outputs/
├── manifest.jsonl                  <- one line per image, all categories
├── celeb/
│   ├── images/
│   │   └── 10037.jpg
│   └── metadata/
│       └── 10037.json
├── object/
│   ├── images/
│   │   └── bicycle_final/111476691561_0.JPG
│   └── metadata/
│       └── bicycle_final/111476691561_0.json
├── cuisine/
│   ├── images/
│   │   └── Nigeria/cuisine/egusi/egusi_Q10771593_0.jpg
│   └── metadata/
│       └── Nigeria/cuisine/egusi/egusi_Q10771593_0.json
└── landmark/
    ├── images/
    │   └── Australia/landscapes/Sydney_Opera_House/Sydney_Opera_House_Q45178.jpg
    └── metadata/
        └── Australia/landscapes/Sydney_Opera_House/Sydney_Opera_House_Q45178.json

Run:
    python stage1_pipeline.py --limit-per-category 20   # quick test
    python stage1_pipeline.py                            # full run
"""

import os
import re
import json
import glob
import time
import base64
import shutil
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"

CATEGORY_PATHS = {
    "celeb": "/mnt/data0/teja/research_multiref/RefCompose/benchmarking/celeb/CelebA-HQ-img/*.jpg",
    "object": "/mnt/data0/teja/research_multiref/cultural_dataset/Stanford_Online_Products/*/*.JPG",
    "cuisine": "/mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images/*/cuisine/*/*.jpg",
    "landmark": "/mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images/*/landscapes/*/*.jpg",
}

CATEGORY_SOURCE_ROOTS = {
    "celeb": "/mnt/data0/teja/research_multiref/RefCompose/benchmarking/celeb/CelebA-HQ-img",
    "object": "/mnt/data0/teja/research_multiref/cultural_dataset/Stanford_Online_Products",
    "cuisine": "/mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images",
    "landmark": "/mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images",
}

OUTPUT_ROOT = "outputs"
NUM_THREADS = 16

PROMPT_TEMPLATE = """You are evaluating a reference image for use in a multi-reference image generation dataset.
This image will serve as a visual reference for an object, celebrity, cuisine dish, or landmark.

Analyze the image and respond in the following strict format:

QUALITY: <GOOD or BAD>
REASON: <one short phrase, max 8 words>
CAPTION: <exactly 50 words describing the image>

Quality Guidelines — mark BAD if any apply:
- Image is blurry, low-resolution, or heavily compressed
- Subject is occluded, cropped awkwardly, or not clearly visible
- Multiple conflicting subjects with no clear focal point
- Extreme/unnatural lighting that obscures key details (silhouettes, overexposure, heavy shadows)
- Watermarks, text overlays, or borders covering significant image area
- Object/subject is at an extreme angle making shape/identity ambiguous
- Image contains heavy stylization/filters that distort true appearance

Otherwise mark GOOD.

Caption Guidelines (exactly 50 words):
- Describe the subject's identity/category first
- Describe pose/orientation/scale
- Describe lighting conditions
- Describe background/context and any interactions
- Avoid subjective adjectives like "beautiful" or "nice" — stay descriptive and factual

Image category (for context): {category}
"""

_manifest_lock = threading.Lock()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(image_path, category):
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    b64 = encode_image_b64(image_path)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": PROMPT_TEMPLATE.format(category=category)
                }
            ]
        }
    ]


def parse_response(text):
    quality_match = re.search(r"QUALITY:\s*(GOOD|BAD)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    caption_match = re.search(r"CAPTION:\s*(.+)", text, re.DOTALL)

    return {
        "quality": quality_match.group(1).upper() if quality_match else None,
        "reason": reason_match.group(1).strip().split("\n")[0] if reason_match else None,
        "caption": caption_match.group(1).strip().replace("\n", " ") if caption_match else None,
    }


def get_output_paths(category, source_path):
    """Mirror source subfolder layout under outputs/<category>/."""
    root = CATEGORY_SOURCE_ROOTS[category]
    rel = os.path.relpath(source_path, root)
    rel_dir = os.path.dirname(rel)
    filename = os.path.basename(source_path)
    stem, _ = os.path.splitext(filename)

    img_dir = os.path.join(OUTPUT_ROOT, category, "images", rel_dir)
    meta_dir = os.path.join(OUTPUT_ROOT, category, "metadata", rel_dir)
    saved_image_path = os.path.join(img_dir, filename)
    saved_meta_path = os.path.join(meta_dir, f"{stem}.json")
    record_id = os.path.join(category, rel_dir, stem) if rel_dir != "." else os.path.join(category, stem)

    return img_dir, meta_dir, saved_image_path, saved_meta_path, record_id


# ----------------------------------------------------------------------------
# Core worker
# ----------------------------------------------------------------------------

def process_one(record_id, image_path, category, saved_image_path, saved_meta_path, max_retries=3):
    img_dir = os.path.dirname(saved_image_path)
    meta_dir = os.path.dirname(saved_meta_path)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    result = {
        "id": record_id,
        "category": category,
        "source_path": image_path,
        "saved_image_path": saved_image_path,
        "saved_metadata_path": saved_meta_path,
        "quality": None,
        "reason": None,
        "caption": None,
        "error": None,
    }

    # 1. Copy image into the paired output folder first (so we have it even if VLM call fails)
    try:
        shutil.copy2(image_path, saved_image_path)
    except Exception as e:
        result["error"] = f"copy_failed: {e}"
        with open(saved_meta_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # 2. Call the VLM with retries
    last_err = None
    for attempt in range(max_retries):
        try:
            messages = build_messages(image_path, category)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            raw_text = response.choices[0].message.content
            parsed = parse_response(raw_text)
            result.update(parsed)
            result["raw_response"] = raw_text
            last_err = None
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)

    if last_err:
        result["error"] = last_err

    # 3. Save paired metadata json (same base name as the saved image)
    with open(saved_meta_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ----------------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------------

def run_pipeline(limit_per_category=None, num_threads=NUM_THREADS):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest.jsonl")

    # Build full task list across all categories first
    tasks = []
    for category, pattern in CATEGORY_PATHS.items():
        paths = sorted(glob.glob(pattern))
        if limit_per_category:
            paths = paths[:limit_per_category]
        for p in paths:
            img_dir, meta_dir, saved_image_path, saved_meta_path, record_id = get_output_paths(category, p)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)
            tasks.append((record_id, p, category, saved_image_path, saved_meta_path))

    print(f"Total images to process: {len(tasks)} across {len(CATEGORY_PATHS)} categories")
    print(f"Using {num_threads} threads")

    start = time.time()
    counts = {"GOOD": 0, "BAD": 0, "ERROR": 0}

    with open(manifest_path, "w") as manifest_f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_one, rid, path, cat, saved_img, saved_meta): rid
                for rid, path, cat, saved_img, saved_meta in tasks
            }
            with tqdm(total=len(tasks), desc="Captioning", unit="img") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    with _manifest_lock:
                        manifest_f.write(json.dumps(result) + "\n")
                        manifest_f.flush()

                    if result.get("error"):
                        counts["ERROR"] += 1
                    elif result.get("quality") == "GOOD":
                        counts["GOOD"] += 1
                    elif result.get("quality") == "BAD":
                        counts["BAD"] += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        GOOD=counts["GOOD"],
                        BAD=counts["BAD"],
                        ERR=counts["ERROR"],
                        refresh=False,
                    )

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"GOOD={counts['GOOD']}  BAD={counts['BAD']}  ERROR={counts['ERROR']}")
    print(f"Manifest: {manifest_path}")
    print(f"Per-category paired data: {OUTPUT_ROOT}/<category>/images/<source-subdirs>/ + /metadata/<source-subdirs>/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-category", type=int, default=None,
                         help="Cap number of images per category (for quick test runs)")
    parser.add_argument("--threads", type=int, default=NUM_THREADS,
                         help="Number of worker threads (default 8)")
    args = parser.parse_args()

    run_pipeline(limit_per_category=args.limit_per_category, num_threads=args.threads)