#!/usr/bin/env python3
"""
flux2_imagegen.py

Reads theme prompts from manifest.jsonl (v2 theme_prompt_composer output) and
generates one FLUX.2 image per manifest row using each sample's combined_prompt.

Designed to be launched once PER GPU. Each launch is told:
  --shard_id      which shard this process handles (0-indexed)
  --num_shards    total number of shards/GPUs (default: 6)

Samples are assigned via `index % num_shards == shard_id`, so you can safely
run 6 independent processes in parallel and they will never touch the same sample.

Typical usage (6 GPUs, one process per GPU):

    # shard 0 on GPU 1, shards 1-5 on GPUs 2-6
    CUDA_VISIBLE_DEVICES=1 python flux2_imagegen.py --shard_id 0 --num_shards 6
    CUDA_VISIBLE_DEVICES=2 python flux2_imagegen.py --shard_id 1 --num_shards 6
    CUDA_VISIBLE_DEVICES=3 python flux2_imagegen.py --shard_id 2 --num_shards 6
    CUDA_VISIBLE_DEVICES=4 python flux2_imagegen.py --shard_id 3 --num_shards 6
    CUDA_VISIBLE_DEVICES=5 python flux2_imagegen.py --shard_id 4 --num_shards 6
    CUDA_VISIBLE_DEVICES=6 python flux2_imagegen.py --shard_id 5 --num_shards 6

    # Or launch all 6 at once:
    bash run_flux2_imagegen.sh

    # Smaller GPUs:
    CUDA_VISIBLE_DEVICES=1 python flux2_imagegen.py --shard_id 0 --num_shards 6 --quantized
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "outputs/theme_prompts/manifest.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/theme_images"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate FLUX.2 images for every row in the theme_prompts manifest."
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Path to theme_prompts manifest.jsonl.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write generated images + sidecar metadata.",
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="0-indexed shard/GPU id this process handles.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=6,
        help="Total number of shards/GPUs (default: 6).",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.2-dev",
        help="HF model id or local path for Flux2Pipeline. Use "
        "'diffusers/FLUX.2-dev-bnb-4bit' with --quantized for consumer GPUs.",
    )
    p.add_argument(
        "--quantized",
        action="store_true",
        help="Load the 4-bit (bnb) quantized transformer + text encoder "
        "instead of full bf16. Automatically enables CPU offload.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string. Use with CUDA_VISIBLE_DEVICES when launching.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="FLUX.2 default guidance is 4.0.",
    )
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Per-sample seed is derived from this + sample id.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples this shard will process (smoke testing).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output image already exists (default: True).",
    )
    p.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    p.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable model CPU offload (use if you hit OOM). Auto-enabled when --quantized.",
    )
    return p.parse_args()


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


def sample_seed(base_seed, sample_id):
    return (base_seed + sum(ord(c) for c in sample_id)) % (2**31 - 1)


def build_pipeline(args):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.quantized:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration

        repo_id = args.model_id if "bnb-4bit" in args.model_id else "diffusers/FLUX.2-dev-bnb-4bit"
        print(f"[shard {args.shard_id}] Loading quantized FLUX.2 ({repo_id}) ...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        pipe = Flux2Pipeline.from_pretrained(
            repo_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
        )
        if args.enable_cpu_offload or args.quantized:
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


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / "images"
    meta_out_dir = out_dir / "metadata"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.shard_id}/{args.num_shards}] Loading manifest: {manifest_path}")
    all_samples = load_manifest(manifest_path)
    all_samples.sort(key=lambda s: s.get("id", ""))
    print(f"[shard {args.shard_id}/{args.num_shards}] Total samples in manifest: {len(all_samples)}")

    my_samples = shard_samples(all_samples, args.shard_id, args.num_shards)
    if args.limit is not None:
        my_samples = my_samples[: args.limit]
    print(f"[shard {args.shard_id}/{args.num_shards}] Samples assigned to this shard: {len(my_samples)}")

    pipe = build_pipeline(args)

    n_done, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()

    for sample in my_samples:
        sample_id = sample.get("id", "unknown")
        theme = sample.get("theme", "unknown")
        prompt = (sample.get("combined_prompt") or "").strip()

        out_img_path = img_out_dir / f"{sample_id}.png"
        out_meta_path = meta_out_dir / f"{sample_id}.json"

        if not prompt:
            print(f"[shard {args.shard_id}] [{sample_id}] SKIP: missing combined_prompt")
            n_failed += 1
            continue

        if args.skip_existing and out_img_path.exists():
            n_skipped += 1
            continue

        try:
            seed = sample_seed(args.seed, sample_id)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            result = pipe(
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
                        "theme": theme,
                        "num_objects": sample.get("num_objects"),
                        "objects": sample.get("objects", []),
                        "object_details": sample.get("object_details", []),
                        "prompt": prompt,
                        "structured_prompt": sample.get("structured_prompt"),
                        "seed": seed,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": args.num_inference_steps,
                        "gen_width": args.width,
                        "gen_height": args.height,
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
                print(
                    f"[shard {args.shard_id}] progress: {n_done} done, "
                    f"{n_skipped} skipped, {n_failed} failed ({rate:.3f} img/s)"
                )

        except Exception as e:
            n_failed += 1
            print(f"[shard {args.shard_id}] [{sample_id}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    elapsed = time.time() - t_start
    print(
        f"[shard {args.shard_id}/{args.num_shards}] DONE. "
        f"generated={n_done} skipped={n_skipped} failed={n_failed} "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
