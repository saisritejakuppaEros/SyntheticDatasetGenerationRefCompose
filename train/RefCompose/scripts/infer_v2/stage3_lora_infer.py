#!/usr/bin/env python3
"""
Stage 3 (infer_v2): FLUX.2 + canvas/depth LoRA — heritage interior pipeline.

Defaults (override with flags or env):
  - Canvas: ``generated_image/black_canvas.jpg``
  - Depth:  ``generated_image/image_gen_depth.png`` (from ``stage2_depth.py``)
  - Prompt: ``generated_image/image_gen.txt``

Canvas and depth are cover-cropped to 1280×720 (aspect preserved, edges trimmed — no letterbox padding).
Depth is degraded by default to match training; use ``--no-canvas_depth_coarse_augment`` for the
sharp DA3 depth map.

Quick start::

    cd infer_v2
    CUDA_VISIBLE_DEVICES=0 ./run_lora.sh

    python stage3_lora_infer.py --no-canvas_depth_coarse_augment --lora_first_depth_steps 4
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from PIL import Image
from safetensors.torch import load_file
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

_INFER_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _INFER_DIR.parent / "train"
_GEN_DIR = _INFER_DIR / "methodology_image"
for _p in (_TRAIN_DIR, _INFER_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.canvas_dataset import (  # noqa: E402
    coarse_degrade_depth_rgb_pil,
    load_depth_image_as_rgb_pil,
    pil_to_model_tensor,
)
from src.flux2_dataloader_validation import _denoise_one  # noqa: E402
from src.jsonl_datasets import multiple_16  # noqa: E402
from src.layers_flux2 import MultiDoubleStreamBlockFlux2LoraProcessor, MultiSingleStreamBlockFlux2LoraProcessor  # noqa: E402
from src.prompt_helper import encode_prompts_flux2  # noqa: E402
from src.transformer_flux import FluxTransformer2DModel  # noqa: E402

_DEFAULT_FLUX2 = os.environ.get(
    "FLUX2_MODEL",
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)
_DEFAULT_LORA = os.environ.get(
    "LORA_PATH",
    # "/mnt/data0/teja/research_multiref/lora_training_v2/train/output/lora_checkpoints_v2/checkpoint-11000/lora.safetensors",
    "/mnt/data0/teja/research_multiref/v2_results/output/lora_checkpoints_v2/checkpoint-3000/lora.safetensors"
)
_DEFAULT_CANVAS = str(_GEN_DIR / "black_canvas.jpg")
_DEFAULT_DEPTH = str(_GEN_DIR / "image_gen_depth.png")
# _DEFAULT_DEPTH = str(_GEN_DIR / "black_canvas_depth.png")
_DEFAULT_PROMPT_FILE = _GEN_DIR / "image_gen.txt"

# Training / inference unified frame (cover crop — no letterbox padding).
UNIFIED_WIDTH = 1280
UNIFIED_HEIGHT = 720


def print_lora_checkpoint_before_load(lora_path: Path | str, *, prefix: str = "") -> Path:
    """Log resolved LoRA checkpoint path (and siblings) before loading weights."""
    p = Path(lora_path).expanduser().resolve()
    tag = f"{prefix} " if prefix else ""
    print(f"{tag}LoRA checkpoint (loading): {p}", flush=True)
    if p.is_file():
        st = p.stat()
        print(
            f"{tag}  file: {p.name}  size={st.st_size / (1024 * 1024):.2f} MiB",
            flush=True,
        )
        ckpt_dir = p.parent
        if ckpt_dir.name.startswith("checkpoint-"):
            out_dir = ckpt_dir.parent
            if out_dir.is_dir():
                siblings = sorted(
                    d.name
                    for d in out_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")
                )
                if siblings:
                    print(f"{tag}  checkpoint dir: {out_dir}", flush=True)
                    print(f"{tag}  available: {', '.join(siblings)}", flush=True)
                    print(f"{tag}  selected: {ckpt_dir.name}", flush=True)
    return p


def _resize_cover_aligned(
    images: list[Image.Image],
    out_w: int,
    out_h: int,
    *,
    ref_index: int = 0,
) -> list[Image.Image]:
    """Scale-to-cover then center-crop to ``out_w``×``out_h``; same crop box for every image (no black bars)."""
    ref = images[ref_index].convert("RGB")
    iw, ih = ref.size
    scale = max(out_w / iw, out_h / ih)
    ref_nw = max(1, int(round(iw * scale)))
    ref_nh = max(1, int(round(ih * scale)))
    left = max(0, (ref_nw - out_w) // 2)
    top = max(0, (ref_nh - out_h) // 2)

    out: list[Image.Image] = []
    for im in images:
        im = im.convert("RGB")
        riw, rih = im.size
        rnw = max(1, int(round(riw * scale)))
        rnh = max(1, int(round(rih * scale)))
        resized = im.resize((rnw, rnh), Image.Resampling.LANCZOS)
        out.append(resized.crop((left, top, left + out_w, top + out_h)))
    return out


def _attach_lora(
    transformer: FluxTransformer2DModel,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    lora_num: int,
    ranks: list[int],
    network_alphas: list[int],
    unified_w: int,
    unified_h: int,
) -> list[float]:
    dim = transformer.inner_dim
    sa0 = transformer.single_transformer_blocks[0].attn
    s_inner, s_mlp_h, s_mlp_mf = sa0.inner_dim, sa0.mlp_hidden_dim, sa0.mlp_mult_factor
    double_blocks_idx = list(range(len(transformer.transformer_blocks)))
    single_blocks_idx = list(range(len(transformer.single_transformer_blocks)))
    lora_w = [1.0 for _ in range(lora_num)]
    lora_cond_w, lora_cond_h = unified_w, unified_h

    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockFlux2LoraProcessor(
                dim=dim,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=lora_num,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockFlux2LoraProcessor(
                dim=dim,
                inner_dim=s_inner,
                mlp_hidden_dim=s_mlp_h,
                mlp_mult_factor=s_mlp_mf,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=lora_num,
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)
    return lora_w


def _lora_interval_w_canvas_depth(
    frac: float,
    *,
    t1: float,
    t2: float,
    t3: float,
    tail: float,
) -> tuple[float, float]:
    """Map denoise progress ``frac`` in [0, 1] to (canvas/structure, depth) LoRA weights.

    LoRA index 0 = canvas, 1 = depth. Early: depth-forward but canvas 0.5/0.98. Mid: canvas to 1.0 by t2, depth
    to ~0.2. t2..t3: full canvas, depth 0.2 to 0.1. For ``f > t3``, both scale toward ``tail``.
    """
    f = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else float(frac)

    def lerp_on(u: float, a: float, b: float, va: float, vb: float) -> float:
        if a >= b - 1e-8:
            return vb
        if u <= a:
            return va
        if u >= b:
            return vb
        t = (u - a) / (b - a)
        return va + t * (vb - va)

    if t1 >= t2 or t2 >= t3 or t3 > 1.0 or t1 <= 0.0:
        return 1.0, 1.0

    # Appearance (canvas) is weighted higher than the original 0.2/0.95/1.0 schedule:
    # earlier ramp to 1.0, slightly less extreme depth-dominant early.
    if f < t1:
        w_s, w_d = 0.5, 0.98
    elif f < t2:
        w_s = lerp_on(f, t1, t2, 0.5, 1.0)
        w_d = lerp_on(f, t1, t2, 0.98, 0.2)
    elif f <= t3:
        w_s = 1.0
        w_d = lerp_on(f, t2, t3, 0.2, 0.1)
    else:
        w_s0, w_d0 = 1.0, 0.1
        m = 1.0
        if tail < 1.0:
            span = max(1.0 - t3, 1e-6)
            m = 1.0 - (1.0 - tail) * (f - t3) / span
        w_s, w_d = w_s0 * m, w_d0 * m
    return w_s, w_d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=_DEFAULT_FLUX2)
    p.add_argument("--lora_path", type=str, default=_DEFAULT_LORA)
    p.add_argument("--canvas_image", type=str, default=_DEFAULT_CANVAS)
    p.add_argument("--depth_image", type=str, default=_DEFAULT_DEPTH)
    p.add_argument(
        "--canvas_depth_coarse_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Match training: randomly degrade resized depth (Gaussian blur, noise, smoothed patch dropout) "
            "before VAE encode. Use --no-canvas_depth_coarse_augment for a clean depth map."
        ),
    )
    p.add_argument(
        "--canvas_depth_coarse_blur_prob",
        type=float,
        default=0.7,
        help="Per forward pass: probability to apply heavy blur to depth (same default as train.py).",
    )
    p.add_argument(
        "--canvas_depth_coarse_noise_prob",
        type=float,
        default=0.4,
        help="Per forward pass: probability to add Gaussian noise to depth.",
    )
    p.add_argument(
        "--canvas_depth_coarse_patch_prob",
        type=float,
        default=0.3,
        help="Per forward pass: probability for smoothed patch dropout on depth.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        metavar="TEXT",
        help="Scene prompt text. If omitted, reads --prompt-file.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=_DEFAULT_PROMPT_FILE,
        help="Prompt text file (default: generated_image/image_gen.txt).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_INFER_DIR / "op" / "image_gen_lora.png"),
    )
    p.add_argument(
        "--cond_variants",
        action="store_true",
        help="Write three images (suffix _depth_black, _canvas_black, _both): black depth+real canvas, black canvas+real depth, and both real (same seed).",
    )
    p.add_argument(
        "--lora_interval_schedule",
        action="store_true",
        help=(
            "Per-denoise-step LoRA mix: 0..t1 depth strong / structure weak; t1..t2 crossfade; t2..t3 structure high / depth low; "
            "after t3 both scale toward --lora_interval_tail (set to 1.0 to skip). Requires --lora_num 2. "
            "If --lora_first_depth_steps > 0, that two-phase block overrides this curve."
        ),
    )
    p.add_argument(
        "--lora_interval_t1",
        type=float,
        default=0.25,
        metavar="F",
        help="Interval schedule: end of early segment (default 0.25).",
    )
    p.add_argument(
        "--lora_interval_t2",
        type=float,
        default=0.70,
        metavar="F",
        help="End of crossfade (default 0.70).",
    )
    p.add_argument(
        "--lora_interval_t3",
        type=float,
        default=0.90,
        metavar="F",
        help="Start of final tail scaling (default 0.90).",
    )
    p.add_argument(
        "--lora_interval_tail",
        type=float,
        default=0.85,
        metavar="F",
        help="At progress=1.0, both LoRAs scale to this (relative to 1,1 at t3+). 1.0 = no last-segment pull-back.",
    )
    p.add_argument(
        "--lora_interval_appearance_strength",
        type=float,
        default=1.0,
        metavar="F",
        help="Extra multiplier on the canvas/appearance (LoRA0) weight after the interval curve; clamped to 1.5. 1.0 = use curve as-is.",
    )
    p.add_argument(
        "--lora_first_depth_steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If N>0, use a simple two-phase mix (requires --lora_num 2): first N denoise steps depth LoRA strong / canvas low; "
            "remaining steps canvas strong / depth low. N is a step count, not a fraction. "
            "When N>0 this overrides the t1/t2/t3 piecewise curve. Example: --lora_first_depth_steps 4."
        ),
    )
    p.add_argument("--unified_width", type=int, default=UNIFIED_WIDTH)
    p.add_argument("--unified_height", type=int, default=UNIFIED_HEIGHT)
    p.add_argument("--lora_num", type=int, default=2)
    p.add_argument("--ranks", type=int, nargs="+", default=[32])
    p.add_argument("--network_alphas", type=int, nargs="+", default=[32])
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=None,
        metavar="G",
        help=(
            "Run multiple guidance values in one process (model loaded once). "
            "Outputs use --output stem with _gs<G> suffix per scale. "
            "Overrides --guidance_scale when set."
        ),
    )
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target_init",
        type=str,
        default="black",
        choices=["black", "canvas"],
        help="Branch used only for latent **shape** (noise is random): black RGB or same as canvas after cover resize.",
    )
    ns = p.parse_args()
    lora_limited_schedule = bool(ns.lora_interval_schedule) or int(ns.lora_first_depth_steps) > 0
    if lora_limited_schedule:
        if int(ns.lora_num) != 2:
            raise SystemExit("LoRA step mixing requires --lora_num 2 (LoRA0=canvas, LoRA1=depth).")
    if ns.lora_interval_schedule and int(ns.lora_first_depth_steps) <= 0:
        t1, t2, t3i = float(ns.lora_interval_t1), float(ns.lora_interval_t2), float(ns.lora_interval_t3)
        if t1 >= t2 or t2 >= t3i or t3i > 1.0 or t1 <= 0.0:
            raise SystemExit("Require 0 < lora_interval_t1 < lora_interval_t2 < lora_interval_t3 <= 1.")

    tw = multiple_16(ns.unified_width)
    th = multiple_16(ns.unified_height)

    scene_prompt = (ns.prompt or "").strip()
    if not scene_prompt:
        prompt_path = Path(ns.prompt_file).expanduser().resolve()
        if not prompt_path.is_file():
            raise SystemExit(f"prompt file not found: {prompt_path}")
        scene_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not scene_prompt:
        raise SystemExit("Empty prompt: pass --prompt or a non-empty --prompt-file.")

    lora_path = print_lora_checkpoint_before_load(ns.lora_path)
    if not lora_path.is_file():
        raise SystemExit(f"lora_path not found: {lora_path}")

    model_path = Path(ns.pretrained_model_name_or_path).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"pretrained_model_name_or_path not a directory: {model_path}")

    canvas_path = Path(ns.canvas_image).expanduser().resolve()
    depth_path = Path(ns.depth_image).expanduser().resolve()
    if not canvas_path.is_file():
        raise SystemExit(f"canvas_image not found: {canvas_path}")
    if not depth_path.is_file():
        raise SystemExit(f"depth_image not found: {depth_path}")

    print(f"Canvas: {canvas_path}")
    print(f"Depth:  {depth_path}")
    print(f"Prompt: {scene_prompt[:120]}{'…' if len(scene_prompt) > 120 else ''}")

    canvas_pil = Image.open(canvas_path).convert("RGB")
    depth_pil = load_depth_image_as_rgb_pil(depth_path)

    print(
        f"Resizing to {tw}×{th} (cover crop, aligned, no letterbox): "
        f"canvas {canvas_pil.size} depth {depth_pil.size}",
        flush=True,
    )
    # Align crop to depth map so structure matches the RGB/depth source frame.
    canvas_u, depth_u = _resize_cover_aligned(
        [canvas_pil, depth_pil], tw, th, ref_index=1
    )
    assert canvas_u.size == (tw, th) and depth_u.size == (tw, th)

    if ns.canvas_depth_coarse_augment:
        depth_u = coarse_degrade_depth_rgb_pil(depth_u, ns)

    if ns.target_init == "canvas":
        target_u = canvas_u.copy()
    else:
        target_u = Image.new("RGB", (tw, th), (0, 0, 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    ranks = list(ns.ranks)
    alphas = list(ns.network_alphas)
    while len(ranks) < ns.lora_num:
        ranks.append(ranks[-1] if ranks else 32)
    while len(alphas) < ns.lora_num:
        alphas.append(alphas[-1] if alphas else 32)
    ranks = ranks[: ns.lora_num]
    alphas = alphas[: ns.lora_num]

    print(f"Device={device} dtype={weight_dtype} unified={tw}×{th} lora_num={ns.lora_num}")
    print(f"Loading tokenizer from {model_path / 'tokenizer'} …")
    tokenizer = PixtralProcessor.from_pretrained(str(model_path / "tokenizer"))
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        str(model_path), subfolder="text_encoder"
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_path), subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae")
    base = Flux2Transformer2DModelBase.from_pretrained(str(model_path), subfolder="transformer")
    transformer = FluxTransformer2DModel.from_config(base.config)
    transformer.load_state_dict(base.state_dict(), strict=True)
    del base

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)

    lora_w = _attach_lora(
        transformer,
        device=device,
        weight_dtype=weight_dtype,
        lora_num=ns.lora_num,
        ranks=ranks,
        network_alphas=alphas,
        unified_w=tw,
        unified_h=th,
    )

    lora_sd = load_file(str(lora_path))
    inc = transformer.load_state_dict(lora_sd, strict=False)
    unexpected = getattr(inc, "unexpected_keys", inc[1] if isinstance(inc, tuple) else [])
    print(f"Merged LoRA state dict ({len(lora_sd)} tensors from {lora_path.name}).")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    if lora_limited_schedule:
        if int(ns.lora_first_depth_steps) > 0:
            print(
                f"LoRA block schedule: first {int(ns.lora_first_depth_steps)} step(s) depth-LoRA priority, "
                f"then canvas/appearance-LoRA priority; appearance_strength={ns.lora_interval_appearance_strength}.",
            )
        if ns.lora_interval_schedule and int(ns.lora_first_depth_steps) <= 0:
            print(
                f"LoRA interval schedule: t1={ns.lora_interval_t1} t2={ns.lora_interval_t2} "
                f"t3={ns.lora_interval_t3} tail@1.0={ns.lora_interval_tail} "
                f"appearance_strength={ns.lora_interval_appearance_strength} (canvas, depth weighter each step).",
            )

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    black_u = Image.new("RGB", (tw, th), (0, 0, 0))
    black_t = pil_to_model_tensor(black_u).unsqueeze(0)
    pv = pil_to_model_tensor(target_u).unsqueeze(0)
    subj_full = pil_to_model_tensor(canvas_u).unsqueeze(0)
    cond_full = pil_to_model_tensor(depth_u).unsqueeze(0)

    pe, tid = encode_prompts_flux2(
        text_encoder,
        tokenizer,
        [scene_prompt],
        device,
        ns.max_sequence_length,
        weight_dtype,
        tuple(ns.text_encoder_out_layers),
    )

    def _lora_weighter(step_i: int, n_steps: int) -> None:
        ast = min(max(float(ns.lora_interval_appearance_strength), 0.0), 1.5)
        n_depth = int(ns.lora_first_depth_steps)
        if n_depth > 0:
            if step_i < n_depth:
                wc, wd = 0.2, 1.0
            else:
                wc, wd = 1.0, 0.1
            lora_w[0], lora_w[1] = wc * ast, wd
            return
        denom = max(1, n_steps - 1)
        frac = float(step_i) / float(denom)
        wc, wd = _lora_interval_w_canvas_depth(
            frac,
            t1=float(ns.lora_interval_t1),
            t2=float(ns.lora_interval_t2),
            t3=float(ns.lora_interval_t3),
            tail=float(ns.lora_interval_tail),
        )
        lora_w[0], lora_w[1] = wc * ast, wd

    def _run_denoise(subj: torch.Tensor, cond: torch.Tensor) -> Image.Image:
        gen = torch.Generator(device=device).manual_seed(int(ns.seed))
        with torch.inference_mode():
            return _denoise_one(
                vae=vae,
                transformer=transformer,
                scheduler_template=noise_scheduler_copy,
                pixel_values=pv,
                subject_pixel_values=subj,
                cond_pixel_values=cond,
                prompt_embeds=pe,
                text_ids=tid,
                guidance_scale=float(ns.guidance_scale),
                weight_dtype=weight_dtype,
                device=device,
                num_inference_steps=int(ns.num_inference_steps),
                generator=gen,
                lora_weighter=_lora_weighter if lora_limited_schedule else None,
            )

    out_base = Path(ns.output).expanduser().resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    guidance_scales = (
        [float(g) for g in ns.guidance_scales]
        if ns.guidance_scales
        else [float(ns.guidance_scale)]
    )
    if len(guidance_scales) > 1:
        print(
            f"Guidance sweep: {len(guidance_scales)} values, model loaded once → {guidance_scales}",
            flush=True,
        )

    def _output_path_for_scale(scale: float) -> Path:
        if len(guidance_scales) == 1:
            return out_base
        scale_label = f"{scale:g}".replace(".", "p")
        return out_base.with_name(f"{out_base.stem}_gs{scale_label}{out_base.suffix}")

    if ns.cond_variants:
        variants: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            ("depth_black", subj_full, black_t),
            ("canvas_black", black_t, cond_full),
            ("both", subj_full, cond_full),
        ]
        for scale in guidance_scales:
            ns.guidance_scale = scale
            scale_label = "" if len(guidance_scales) == 1 else f"_gs{f'{scale:g}'.replace('.', 'p')}"
            for name, subj, cond in variants:
                out_pil = _run_denoise(subj, cond)
                outp = out_base.with_name(f"{out_base.stem}{scale_label}_{name}{out_base.suffix}")
                out_pil.save(outp)
                print(f"Saved (gs={scale:g}, {name}) → {outp}")
                with open(outp.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write(scene_prompt)
    else:
        for scale in guidance_scales:
            ns.guidance_scale = scale
            outp = _output_path_for_scale(scale)
            out_pil = _run_denoise(subj_full, cond_full)
            out_pil.save(outp)
            print(f"Saved (guidance_scale={scale:g}) → {outp}")
            with open(outp.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(scene_prompt)


if __name__ == "__main__":
    main()
