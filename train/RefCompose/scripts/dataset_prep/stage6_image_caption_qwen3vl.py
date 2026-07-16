#!/usr/bin/env python3
"""
Stage 6: caption every image under ``output/images`` (or ``--image-dir``) with Qwen3-VL,
saving **one standalone text-to-image prompt per file** (~100 words): dense visual
description only—no framing like ``The image consists of ...``, so each ``.txt`` can be fed
directly to an image generator. By default snapshots go under ``dataset_preparation/hf_models``
(see ``--cache-dir``).

Requires a recent Transformers build that includes ``Qwen3VLForConditionalGeneration``
(see model card). Example::

    pip install git+https://github.com/huggingface/transformers

Run::

    python stage6_image_caption_qwen3vl.py

Optional::

    python stage6_image_caption_qwen3vl.py --flash-attn2 --resume

**Multi-host / multi-session:** load **one** model per process. Use ``CUDA_VISIBLE_DEVICES``
and disjoint ``--index-start`` / ``--index-end`` (half-open slice over the sorted image list).
Print suggested ranges::

    python stage6_image_caption_qwen3vl.py --print-shard-ranges 3

Then run three terminals with ``run_stage6_three_gpus.sh`` (edit variables if needed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "output" / "images"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "image_captions"
# Keep HF snapshots next to this script instead of ~/.cache/huggingface/hub.
DEFAULT_CACHE_DIR = SCRIPT_DIR / "hf_models"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

CAPTION_INSTRUCTION = (
    "Write a single dense text-to-image prompt of approximately 100 words that could "
    "regenerate this scene: subject matter, composition, camera angle, lighting, colors, "
    "materials, mood, and style. Output only that prompt as one paragraph. "
    "Do not use meta-phrases such as 'The image', 'This image', 'The photo', "
    "'The picture shows', 'The scene depicts', or 'consists of'—start directly with "
    "what is visible (e.g. subjects and setting)."
)

# Strip if the model still echoes tutorial-style openers (so saved files stay gen-ready).
_BAD_OPENERS = (
    "the image consists of",
    "the image shows",
    "the image depicts",
    "the image features",
    "this image shows",
    "this image depicts",
    "this photograph shows",
    "the photograph shows",
    "the picture shows",
    "this picture shows",
    "the scene shows",
    "the scene depicts",
)


def _list_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))
    # stable, dedupe (case variants on case-insensitive FS)
    uniq: dict[str, Path] = {}
    for p in sorted(paths):
        try:
            k = str(p.resolve())
        except OSError:
            k = str(p)
        uniq[k] = p
    return sorted(uniq.values(), key=lambda x: str(x).lower())


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _trim_to_words(text: str, max_words: int) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text.strip()
    s = " ".join(parts[:max_words]).strip()
    if not s.endswith((".", "!", "?", "…")):
        s += "."
    return s


def _clean_caption(text: str) -> str:
    """Drop leading meta-phrases so the text reads as a standalone generation prompt."""
    t = text.strip()
    while t:
        low = t.lower()
        stripped = False
        for opener in sorted(_BAD_OPENERS, key=len, reverse=True):
            if low.startswith(opener):
                t = t[len(opener) :].lstrip(" \t\n\r:–—-•")
                stripped = True
                break
        if not stripped:
            break
    return t.strip()


def _shard_ranges(n: int, k: int) -> list[tuple[int, int]]:
    """Half-open slices [start, end) that partition range(n) into k contiguous shards."""
    if k < 1:
        raise ValueError("k >= 1")
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        rem_shards = k - i
        rem_n = n - start
        sz = (rem_n + rem_shards - 1) // rem_shards
        end = min(n, start + sz)
        ranges.append((start, end))
        start = end
    return ranges


def _caption_one(
    im_path: Path,
    *,
    torch,
    model,
    processor,
    device,
    max_new_tokens: int,
    target_words: int,
) -> str:
    image = Image.open(im_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_INSTRUCTION},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    caption = _clean_caption(raw)
    tw = target_words
    if tw > 0 and _word_count(caption) > tw + 15:
        caption = _trim_to_words(caption, tw)
    return caption.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL captions for every image under a directory (~100 words)."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Root folder of images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write .txt captions (mirrors subfolders, default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="HF model id for Qwen3-VL.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "HF hub snapshot directory for this script’s model and processor downloads "
            f"(default: {DEFAULT_CACHE_DIR}, alongside this script—not ~/.cache/huggingface)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Generation budget (100 English words ~= 130–180 tokens).",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=100,
        help="Soft target length; captions longer than this are trimmed by words.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, only process this many images (after sorting paths).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose output .txt already exists.",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        help="Use attn_implementation='flash_attention_2' (must be installed).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Torch dtype for weights (default auto).",
    )
    parser.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="First index into the sorted image list (inclusive). Use with --index-end to shard.",
    )
    parser.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="End index into the sorted list (exclusive). Default: len(images).",
    )
    parser.add_argument(
        "--print-shard-ranges",
        type=int,
        metavar="N",
        nargs="?",
        const=3,
        default=None,
        help="Print N disjoint [start,end) index ranges over the sorted list and exit (default N=3).",
    )
    args = parser.parse_args()

    image_dir: Path = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir: Path = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir)

    paths_all = _list_images(image_dir)
    if args.max_images > 0:
        paths_all = paths_all[: args.max_images]
    n_all = len(paths_all)

    if args.print_shard_ranges is not None:
        k = args.print_shard_ranges
        if k < 1:
            raise SystemExit("--print-shard-ranges N requires N >= 1")
        print(f"sorted_image_count={n_all}", flush=True)
        print(f"image_dir={image_dir}", flush=True)
        if n_all == 0:
            print("shard_0: (no images — nothing to split)", flush=True)
            return
        for i, (a, b) in enumerate(_shard_ranges(n_all, k)):
            cnt = b - a
            print(
                f"shard_{i}: --index-start {a} --index-end {b}   ({cnt} images)",
                flush=True,
            )
        return

    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install torch and a recent transformers "
            "(`pip install git+https://github.com/huggingface/transformers`).\n"
            f"Original error: {e}"
        ) from e

    if n_all == 0:
        raise SystemExit(f"No images found under {image_dir} (extensions: {IMAGE_EXTS}).")

    dtype_kw: str | torch.dtype = args.dtype
    if args.dtype == "auto":
        dtype_kw = "auto"
    elif args.dtype == "bfloat16":
        dtype_kw = torch.bfloat16
    elif args.dtype == "float16":
        dtype_kw = torch.float16
    else:
        dtype_kw = torch.float32

    load_kw: dict = {
        "dtype": dtype_kw,
        "device_map": "auto",
        "cache_dir": cache_dir_str,
    }
    if args.flash_attn2:
        load_kw["attn_implementation"] = "flash_attention_2"

    istart = max(0, args.index_start)
    iend = n_all if args.index_end is None else args.index_end
    if istart > n_all:
        raise SystemExit(f"--index-start {istart} beyond list length {n_all}")
    if iend < istart:
        raise SystemExit(f"--index-end {iend} must be >= --index-start {istart}")
    if iend > n_all:
        print(
            f"Warning: clamping --index-end from {iend} to {n_all} (sorted list length)",
            file=sys.stderr,
        )
        iend = n_all

    paths = paths_all[istart:iend]

    print(f"HF cache_dir (processor + model weights): {cache_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir_str)

    pending: list[tuple[Path, Path]] = []
    for im in paths:
        rel = im.relative_to(image_dir)
        txt_path = (out_dir / rel).with_suffix(".txt")
        if args.resume and txt_path.is_file():
            continue
        pending.append((im, txt_path))

    print(
        f"Shard slice [{istart},{iend}) of {n_all} sorted images; "
        f"{len(paths)} in slice, {len(pending)} to caption "
        f"(resume={args.resume}). Model={args.model!r}",
        flush=True,
    )

    if not pending:
        print(f"Done. Captions under {out_dir}", flush=True)
        return

    print("Loading model…", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **load_kw)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    for im_path, txt_path in tqdm(pending, desc="caption", unit="img"):
        try:
            caption = _caption_one(
                im_path,
                torch=torch,
                model=model,
                processor=processor,
                device=device,
                max_new_tokens=args.max_new_tokens,
                target_words=args.target_words,
            )
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(caption + "\n", encoding="utf-8")
        except Exception as e:
            print(f"skip {im_path}: {e}", file=sys.stderr)

    print(f"Done. Captions under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
