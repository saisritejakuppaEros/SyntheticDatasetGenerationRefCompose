#!/usr/bin/env python3
"""
FLUX.2 text-to-image inference for the Indian heritage interior prompt.

Loads ``Flux2Pipeline`` once and writes PNG(s) under ``outputs/`` by default.
Supports multiple variants with different seeds (same pattern as ``client_flux.py``).

Example::

  CUDA_VISIBLE_DEVICES=0 python flux2_heritage_infer.py

  python flux2_heritage_infer.py --variants 4 --seed 42 -o outputs/heritage_interior.png

  python flux2_heritage_infer.py --prompt-file my_prompt.txt --single

Environment:
  FLUX2_MODEL  — path to FLUX.2-dev snapshot (optional override)
"""

print("running the script")

# from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_FLUX2 = os.environ.get(
    "FLUX2_MODEL",
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)

HERITAGE_INTERIOR_PROMPT = """
Design a refined Indian heritage-inspired interior with highly recognizable and geometrically uniform traditional decor objects: a large polished Brass Urli filled with floating flowers, a tall symmetrical Antique Brass Lamp, a handcrafted Bidriware Vase with silver inlay, a rustic Copper Water Vessel, a flat handwoven Dhurrie rug with geometric patterns, an intricately carved Jaali Partition creating patterned light shadows, an ornate wooden Jharokha frame mounted on the wall, a carved Chettinad Pillar placed architecturally in the room, elegant Cane Furniture with woven textures, and a hanging Temple Bell near the entrance. Maintain clean object visibility and distinct object boundaries for easy recognition and compositing. Use warm ambient lighting, soft sunlight, earthy tones, teakwood textures, sandstone walls, cinematic framing, luxury Indian interior styling, highly realistic materials, photorealistic rendering, centered composition, minimal clutter, architectural digest aesthetic, ultra detailed, sharp focus, 8k interior photography style.""".strip()


def load_flux2_pipeline(
    model_path: str,
    *,
    torch_dtype=torch.bfloat16,
) -> Flux2Pipeline:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"FLUX2 model path is not a directory: {model_path}")
    t0 = time.perf_counter()
    print(f"[flux2] Loading Flux2Pipeline from {model_path!r} (dtype={torch_dtype}) …", flush=True)
    pipe = Flux2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    print(f"[flux2] from_pretrained done in {time.perf_counter() - t0:.1f}s; enabling CPU offload …", flush=True)
    pipe.enable_model_cpu_offload()
    print(f"[flux2] Model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def generate_one(
    pipe: Flux2Pipeline,
    *,
    prompt: str,
    output_path: Path,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))

    cap_preview = (prompt[:200] + "…") if len(prompt) > 200 else prompt
    print(
        f"[flux2] Generating {width}×{height} steps={num_inference_steps} "
        f"guidance={guidance_scale} seed={seed}",
        flush=True,
    )
    print(f"[flux2]   prompt: {cap_preview.replace(chr(10), ' ')!r}", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
    elapsed = time.perf_counter() - t0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    prompt_path = output_path.with_suffix(".txt")
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[flux2] Saved {output_path.resolve()} ({elapsed:.1f}s)", flush=True)
    print(f"[flux2] Wrote prompt → {prompt_path.resolve()}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="FLUX.2 T2I: Indian heritage interior scene.")
    p.add_argument(
        "--model",
        default=_DEFAULT_FLUX2,
        help="Path to FLUX.2-dev snapshot (or set FLUX2_MODEL).",
    )
    p.add_argument(
        "--prompt",
        default="",
        help="Override scene prompt. If omitted, uses built-in heritage interior prompt.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read prompt from file instead of built-in default.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(_SCRIPT_DIR / "outputs" / "heritage_interior.png"),
        help="Output path. For multiple variants: {stem}_varNN_seedS{suffix} in same dir.",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=4,
        metavar="N",
        help="Number of images with different seeds (default: 4).",
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Shortcut for --variants 1 (one file exactly at -o).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed for first variant.")
    p.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Added to seed for each subsequent variant.",
    )
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    if args.single:
        args.variants = 1
    if args.variants < 1:
        print("--variants must be >= 1", file=sys.stderr)
        return 2

    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            print(f"Missing prompt file: {args.prompt_file}", file=sys.stderr)
            return 2
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        prompt = (args.prompt or "").strip() or HERITAGE_INTERIOR_PROMPT

    if not prompt:
        print("Empty prompt.", file=sys.stderr)
        return 2

    out_base = Path(args.output)
    out_dir = out_base.parent
    ext = out_base.suffix or ".png"
    base_stem = out_base.stem or "heritage_interior"

    pipe: Flux2Pipeline | None = None
    try:
        pipe = load_flux2_pipeline(args.model)
        for vi in range(args.variants):
            use_seed = args.seed + vi * args.seed_step
            if args.variants == 1:
                out_path = out_base if out_base.suffix else out_dir / f"{base_stem}{ext}"
            else:
                out_path = out_dir / f"{base_stem}_var{vi + 1:02d}_seed{use_seed}{ext}"

            generate_one(
                pipe,
                prompt=prompt,
                output_path=out_path,
                seed=use_seed,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
            )
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}", file=sys.stderr)
        return 1
    finally:
        if pipe is not None:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
