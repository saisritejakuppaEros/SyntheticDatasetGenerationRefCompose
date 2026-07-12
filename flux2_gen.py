#!/usr/bin/env python3
"""
generate_multiref_images_flux2.py

Reads a manifest.jsonl (stage2_composed style, one JSON object per line) where
each sample has multiple reference images + per-image captions + a
"combined_caption" describing the target composition. Generates one output
image per sample using FLUX.2 (Flux2Pipeline), and saves it to disk.

WHY THIS VERSION EXISTS (vs. the FLUX.1 Kontext collage script):
FLUX.1 Kontext's diffusers pipeline could not reliably accept a *list* of
separate reference images (`image=[img1, img2, ...]`) — depending on the
diffusers version it would either crash during latent packing, or "work" but
visibly reproduce a side-by-side layout if you pre-stitched a collage
yourself, because the model was literally being shown a collage image.

FLUX.2 fixes this properly: `Flux2Pipeline` accepts a genuine list of
separate PIL images and concatenates their *latents* (not their pixels) into
one conditioning sequence. The model treats each image as an independent
reference — no seams, no forced side-by-side composition, no collage
artifacts in the output. So this script no longer does any stitching; it
just loads each reference image and hands the list straight to the pipeline.

FLUX.2 supports up to 10 reference images. You can refer to them in the
prompt either by index ("image 1", "image 2") or by content description
("the mountain valley", "the telescoping pole") — BFL recommends combining
both for best results, which is what build_prompt() below does.

Designed to be launched once PER GPU. Each launch is told:
  --shard_id      which shard this process handles (0-indexed)
  --num_shards    total number of shards/GPUs (e.g. 6)

Samples are assigned to shards via `index % num_shards == shard_id`, so you
can safely run 6 independent processes (one per GPU) in parallel and they
will never touch the same sample.

Typical usage (run this 6 times, once per GPU, e.g. from run_all_gpus.sh):

    CUDA_VISIBLE_DEVICES=0 python generate_multiref_images_flux2.py \
        --manifest /path/to/stage2_composed/manifest.jsonl \
        --images_root /path/to/synthetic_dataset_generation \
        --output_dir /path/to/outputs/stage3_generated_flux2 \
        --shard_id 0 --num_shards 6

Requirements:
    pip install -U "git+https://github.com/huggingface/diffusers"  # FLUX.2 needs diffusers main
    pip install -U transformers accelerate

    Model access: you must accept the gating on the black-forest-labs/FLUX.2-dev
    repo on Hugging Face and run `hf auth login` once per machine.

    Full bf16 FLUX.2-dev needs >80GB VRAM loaded all at once. If your GPUs
    are smaller, use --quantized (loads the bnb-4bit repo + CPU offload)
    instead of raising OOM half way through a shard.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Multi-reference image generation with FLUX.2, sharded across GPUs.")
    p.add_argument("--manifest", type=str, required=True,
                    help="Path to manifest.jsonl (one JSON sample per line).")
    p.add_argument("--images_root", type=str, required=True,
                    help="Root directory that reference_images paths are relative to.")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Where to write generated images + sidecar metadata.")
    p.add_argument("--shard_id", type=int, required=True,
                    help="0-indexed shard/GPU id this process handles.")
    p.add_argument("--num_shards", type=int, default=6,
                    help="Total number of shards/GPUs (default: 6).")
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-dev",
                    help="HF model id or local path for Flux2Pipeline. Use "
                         "'diffusers/FLUX.2-dev-bnb-4bit' with --quantized for "
                         "consumer GPUs.")
    p.add_argument("--quantized", action="store_true",
                    help="Load the 4-bit (bnb) quantized transformer + text encoder "
                         "instead of full bf16. Needed if you don't have >80GB VRAM "
                         "available on a single GPU. Automatically enables CPU offload.")
    p.add_argument("--device", type=str, default="cuda:0",
                    help="Device string. Leave as cuda:0 if you launch each shard "
                         "with CUDA_VISIBLE_DEVICES=<gpu> set externally (recommended).")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--guidance_scale", type=float, default=4.0,
                    help="FLUX.2 default guidance is 4.0 (higher than FLUX.1 Kontext's 2.5).")
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--max_ref_images", type=int, default=3,
                    help="Cap on number of reference images used per sample "
                         "(FLUX.2 supports up to 10, but more refs = more VRAM/time).")
    p.add_argument("--width", type=int, default=1280,
                    help="Output width.")
    p.add_argument("--height", type=int, default=720,
                    help="Output height.")
    p.add_argument("--seed", type=int, default=42,
                    help="Base seed. Actual seed per-sample is derived from this + sample id.")
    p.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of samples this shard will process (smoke testing).")
    p.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip samples whose output image already exists (default: True).")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--enable_cpu_offload", action="store_true",
                    help="Enable model CPU offload (use if you hit OOM). Auto-enabled "
                         "when --quantized is set.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #
def load_manifest(manifest_path):
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at line {line_no}: {e}", file=sys.stderr)
                continue
            samples.append(obj)
    return samples


def shard_samples(samples, shard_id, num_shards):
    return [s for idx, s in enumerate(samples) if idx % num_shards == shard_id]


# --------------------------------------------------------------------------- #
# Reference image loading (NO stitching/collaging — FLUX.2 takes a real list)
# --------------------------------------------------------------------------- #
def resolve_reference_images(sample, images_root, max_ref_images):
    rel_paths = sample.get("reference_images", [])[:max_ref_images]
    images = []
    missing = []
    for rel in rel_paths:
        full_path = os.path.join(images_root, rel)
        if not os.path.isfile(full_path):
            missing.append(full_path)
            continue
        img = Image.open(full_path).convert("RGB")
        images.append(img)
    return images, missing


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_prompt(sample, num_images):
    """
    BFL's own guidance for FLUX.2 multi-reference prompting: refer to each
    reference by BOTH its index ("image 1") and a short natural-language
    description of its content ("the mountain valley"). This is more robust
    than index-only or description-only referencing.
    """
    combined_caption = sample.get("combined_caption", "").strip()
    ref_captions = sample.get("reference_captions", [])[:num_images]

    lines = [combined_caption, ""]

    if num_images > 1:
        lines.append("Reference images:")
        for i, cap in enumerate(ref_captions, start=1):
            short_cap = (cap or "").strip()
            lines.append(f"- Image {i}: {short_cap}")
        lines.append(
            "Combine these references into a single unified, photorealistic scene "
            "matching the description above. Preserve the identity, structure, and "
            "distinguishing features of each subject exactly as shown in its "
            "reference image. Do not merge, swap, or blend features between "
            "different reference subjects, and do not depict the references as "
            "separate panels — render one cohesive shot."
        )
    else:
        cap = (ref_captions[0] if ref_captions else "").strip()
        lines.append(f"Reference image shows: {cap}")
        lines.append(
            "Preserve the identity and distinguishing features of the subject "
            "exactly as shown in the reference image, composited into the scene "
            "described above."
        )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(args):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if args.quantized:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration

        repo_id = args.model_id if "bnb-4bit" in args.model_id else "diffusers/FLUX.2-dev-bnb-4bit"
        print(f"[shard {args.shard_id}] Loading quantized FLUX.2 ({repo_id}) ...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            repo_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="auto",
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=torch_dtype, device_map="auto",
        )
        pipe = Flux2Pipeline.from_pretrained(
            repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype,
        )
        if args.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
    else:
        from diffusers import Flux2Pipeline

        print(f"[shard {args.shard_id}] Loading pipeline {args.model_id} (dtype={args.dtype}) ...")
        pipe = Flux2Pipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)

        if args.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)

    return pipe


# --------------------------------------------------------------------------- #
# Main generation loop
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / "images"
    meta_out_dir = out_dir / "metadata"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.shard_id}/{args.num_shards}] Loading manifest: {args.manifest}")
    all_samples = load_manifest(args.manifest)
    print(f"[shard {args.shard_id}/{args.num_shards}] Total samples in manifest: {len(all_samples)}")

    my_samples = shard_samples(all_samples, args.shard_id, args.num_shards)
    if args.limit is not None:
        my_samples = my_samples[: args.limit]
    print(f"[shard {args.shard_id}/{args.num_shards}] Samples assigned to this shard: {len(my_samples)}")

    pipe = build_pipeline(args)

    n_done, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()

    for i, sample in enumerate(my_samples):
        sample_id = sample.get("id", f"unknown_{i}")
        out_img_path = img_out_dir / f"{sample_id}.png"
        out_meta_path = meta_out_dir / f"{sample_id}.json"

        if args.skip_existing and out_img_path.exists():
            n_skipped += 1
            continue

        try:
            images, missing = resolve_reference_images(sample, args.images_root, args.max_ref_images)
            if missing:
                print(f"[shard {args.shard_id}] [{sample_id}] WARNING missing ref images: {missing}")
            if not images:
                print(f"[shard {args.shard_id}] [{sample_id}] SKIP: no reference images could be loaded.")
                n_failed += 1
                continue

            n_images = len(images)
            prompt = build_prompt(sample, n_images)

            seed_str = "".join(ch for ch in sample_id if ch.isdigit()) or "0"
            sample_seed = (args.seed + int(seed_str)) % (2**31 - 1)
            generator = torch.Generator(device="cpu").manual_seed(sample_seed)

            # KEY CHANGE vs. the old script: pass the raw list of PIL images
            # directly. No collage, no stitching. Flux2Pipeline concatenates
            # their latents internally.
            result = pipe(
                image=images if n_images > 1 else images[0],
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                width=args.width,
                height=args.height,
                generator=generator,
            )
            out_image = result.images[0]
            out_image.save(out_img_path)

            with open(out_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": sample_id,
                        "num_references": n_images,
                        "reference_images": sample.get("reference_images", []),
                        "gen_width": args.width,
                        "gen_height": args.height,
                        "prompt": prompt,
                        "seed": sample_seed,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": args.num_inference_steps,
                        "model_id": args.model_id,
                        "output_image": str(out_img_path),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            n_done += 1
            if n_done % 5 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"[shard {args.shard_id}] progress: {n_done} done, "
                      f"{n_skipped} skipped, {n_failed} failed ({rate:.3f} img/s)")

        except Exception as e:
            n_failed += 1
            print(f"[shard {args.shard_id}] [{sample_id}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    elapsed = time.time() - t_start
    print(f"[shard {args.shard_id}/{args.num_shards}] DONE. "
          f"generated={n_done} skipped={n_skipped} failed={n_failed} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()