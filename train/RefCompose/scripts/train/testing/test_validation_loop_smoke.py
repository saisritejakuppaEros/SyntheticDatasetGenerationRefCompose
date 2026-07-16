#!/usr/bin/env python3
"""
Smoke-test the dataloader validation path (loss + PNGs + TensorBoard), with **dummy LoRA weights**.

This mirrors ``train.py``: load FLUX.2 checkpoint, wrap with ``FluxTransformer2DModel``, attach
``Multi*Flux2LoraProcessor``, then **re-randomize only LoRA parameter tensors** so outputs differ
from a plain base checkpoint without needing a trained ``lora.safetensors``.

Run from ``lora_training/v2/train``::

    python testing/test_validation_loop_smoke.py

Faster / fewer steps::

    python testing/test_validation_loop_smoke.py --validation_inference_steps 4 --validation_num_samples 2

Much faster geometry (smaller H×W → far fewer DiT tokens; still full weights)::

    python testing/test_validation_loop_smoke.py --smoke_fast --validation_inference_steps 4 --validation_num_samples 1

**TensorBoard files:** Hugging Face Accelerate writes under ``<run_root>/tb/<tracker_project_name>/``,
e.g. ``testing/validation_loop_smoke/tb/validation_loop_smoke/events.out.tfevents.*`` — not directly in ``tb/``.
Use ``tensorboard --logdir …/tb`` (TensorBoard scans subfolders).

Use another checkpoint::

    FLUX2_MODEL_DIR=/path/to/FLUX.2-dev python testing/test_validation_loop_smoke.py

TensorBoard (logdir is printed at the end)::

    tensorboard --logdir testing/validation_loop_smoke/tb

Pytest (same full run, gated by env)::

    cd lora_training/v2/train && RUN_FLUX2_VALIDATION_SMOKE=1 pytest testing/test_validation_loop_smoke.py::test_validation_loop_smoke_env_gated -v

Outputs under ``testing/validation_loop_smoke/`` by default:

- ``validation_out/validation_samples/step_XXXXXXX/*.png`` — target, canvas, gen
- ``validation_out/validation_metrics.jsonl`` — one JSON line per validation (always; use if TensorBoard is missing)
- ``tb/validation_loop_smoke/`` — ``events.out.tfevents.*`` (after ``init_trackers``; more scalars/images after validation)

**Rough runtime:** model load ~30–90s on NVMe; validation at **1920×1088** is dominated by **token count**
(thousands of image tokens per forward). Expect **many minutes** on GPU for 2×(4–8) denoise steps unless you use ``--smoke_fast``.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

_TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from src.canvas_dataset import collate_fn_canvas, make_canvas_train_dataset
from src.flux2_dataloader_validation import run_dataloader_validation
from src.jsonl_datasets import multiple_16
from src.layers_flux2 import MultiDoubleStreamBlockFlux2LoraProcessor, MultiSingleStreamBlockFlux2LoraProcessor
from src.transformer_flux import FluxTransformer2DModel

# Defaults aligned with testing/test_dataloader.py
_DEFAULT_MODEL = (
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/"
    "snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
)
_DEFAULT_CSV = (
    "/mnt/data0/teja/research_multiref/dataset_preparation/output/bbox_results/yolo26_detections.csv"
)
_DEFAULT_IMAGE_ROOT = "/mnt/data0/teja/research_multiref/dataset_preparation/output/images"
_DEFAULT_MULTIVIEW = "/mnt/data0/teja/research_multiref/dataset_preparation/output/multiview_out"

_LORA_SUBSTRINGS = ("q_loras", "k_loras", "v_loras", "proj_loras", "qkv_mlp_loras")


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


def _attach_lora(transformer: FluxTransformer2DModel, args: SimpleNamespace, device: torch.device, weight_dtype: torch.dtype) -> None:
    dim = transformer.inner_dim
    sa0 = transformer.single_transformer_blocks[0].attn
    s_inner, s_mlp_h, s_mlp_mf = sa0.inner_dim, sa0.mlp_hidden_dim, sa0.mlp_mult_factor
    double_blocks_idx = list(range(len(transformer.transformer_blocks)))
    single_blocks_idx = list(range(len(transformer.single_transformer_blocks)))
    lora_w = [1.0 for _ in range(args.lora_num)]

    if args.dataset_type == "canvas" and not args.canvas_random_target_resolution:
        lora_cond_w, lora_cond_h = args.unified_train_width, args.unified_train_height
    else:
        lora_cond_w = lora_cond_h = args.cond_size

    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockFlux2LoraProcessor(
                dim=dim,
                ranks=args.ranks,
                network_alphas=args.network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=args.lora_num,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockFlux2LoraProcessor(
                dim=dim,
                inner_dim=s_inner,
                mlp_hidden_dim=s_mlp_h,
                mlp_mult_factor=s_mlp_mf,
                ranks=args.ranks,
                network_alphas=args.network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=lora_cond_w,
                cond_height=lora_cond_h,
                n_loras=args.lora_num,
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)


def _randomize_lora_weights(transformer: FluxTransformer2DModel, std: float = 0.02) -> None:
    with torch.no_grad():
        for name, p in transformer.named_parameters():
            if any(s in name for s in _LORA_SUBSTRINGS):
                p.normal_(0, std)


def _build_canvas_args(ns: argparse.Namespace) -> SimpleNamespace:
    """Namespace with every field ``run_dataloader_validation`` / canvas dataset need."""
    a = SimpleNamespace(
        dataset_type="canvas",
        spatial_column="None",
        csv_path=ns.csv_path,
        canvas_image_root=ns.canvas_image_root,
        canvas_conditioning=ns.canvas_conditioning,
        canvas_bbox_min_side=ns.canvas_bbox_min_side,
        canvas_multiview_match_min_side=ns.canvas_multiview_match_min_side,
        canvas_multiview_dir=ns.canvas_multiview_dir,
        canvas_multiview_prob=ns.canvas_multiview_prob,
        canvas_background=ns.canvas_background,
        canvas_column=ns.canvas_column,
        canvas_target_column=ns.canvas_target_column,
        canvas_prompt_column=ns.canvas_prompt_column,
        cond_size=ns.cond_size,
        noise_size=ns.noise_size,
        unified_train_width=multiple_16(ns.unified_train_width),
        unified_train_height=multiple_16(ns.unified_train_height),
        canvas_unified_resize=ns.canvas_unified_resize,
        canvas_random_target_resolution=False,
        # LoRA
        lora_num=ns.lora_num,
        ranks=ns.ranks,
        network_alphas=ns.network_alphas,
        # Text / loss / validation
        max_sequence_length=ns.max_sequence_length,
        text_encoder_out_layers=ns.text_encoder_out_layers,
        weighting_scheme=ns.weighting_scheme,
        logit_mean=ns.logit_mean,
        logit_std=ns.logit_std,
        mode_scale=ns.mode_scale,
        guidance_scale=ns.guidance_scale,
        validation_num_samples=ns.validation_num_samples,
        validation_inference_steps=ns.validation_inference_steps,
        validation_samples_subdir=ns.validation_samples_subdir,
        validation_canvas_augment=getattr(ns, "validation_canvas_augment", True),
        seed=ns.seed,
        output_dir=str(ns.run_root / "validation_out"),
        # unused but present on real train Namespace
        revision=None,
        variant=None,
    )
    return a


def main() -> None:
    here = Path(__file__).resolve().parent
    default_root = here / "validation_loop_smoke"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=os.environ.get("FLUX2_MODEL_DIR", _DEFAULT_MODEL))
    p.add_argument("--run_root", type=Path, default=default_root, help="Parent dir for TB logs, accelerate project, validation PNGs.")
    p.add_argument("--csv_path", type=str, default=_DEFAULT_CSV)
    p.add_argument("--canvas_image_root", type=str, default=_DEFAULT_IMAGE_ROOT)
    p.add_argument("--canvas_multiview_dir", type=str, default=_DEFAULT_MULTIVIEW)
    p.add_argument("--canvas_conditioning", type=str, default="bbox_multiview", choices=["precomputed", "bbox_multiview"])
    p.add_argument("--canvas_bbox_min_side", type=float, default=0.0)
    p.add_argument("--canvas_multiview_match_min_side", type=float, default=200.0)
    p.add_argument("--canvas_multiview_prob", type=float, default=0.5)
    p.add_argument("--canvas_background", type=str, default="black", choices=["black", "scaled"])
    p.add_argument("--canvas_column", type=str, default="canvas_path")
    p.add_argument("--canvas_target_column", type=str, default="image_path")
    p.add_argument("--canvas_prompt_column", type=str, default="prompt")
    p.add_argument("--cond_size", type=int, default=512)
    p.add_argument("--noise_size", type=int, default=1024)
    p.add_argument("--unified_train_width", type=int, default=1920)
    p.add_argument("--unified_train_height", type=int, default=1080)
    p.add_argument(
        "--canvas_unified_resize",
        type=str,
        default="contain",
        choices=["cover", "contain"],
        help="Match train.py: contain (default) = letterbox; cover = center-crop.",
    )
    p.add_argument("--lora_num", type=int, default=1)
    p.add_argument("--ranks", type=int, nargs="+", default=[128])
    p.add_argument("--network_alphas", type=int, nargs="+", default=[128])
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--weighting_scheme", type=str, default="none")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--validation_num_samples", type=int, default=2)
    p.add_argument("--validation_inference_steps", type=int, default=8, help="Lower = faster smoke test.")
    p.add_argument("--validation_samples_subdir", type=str, default="validation_samples")
    p.add_argument("--dummy_lora_std", type=float, default=0.02, help="Std dev for random LoRA weights.")
    p.add_argument("--global_step", type=int, default=42, help="Step tag for folders and TensorBoard.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smoke_fast",
        action="store_true",
        help="Use 768×448 unified canvas/target (multiple of 16) instead of 1920×1088 — much faster smoke test.",
    )
    ns = p.parse_args()

    if ns.smoke_fast:
        ns.unified_train_width = 768
        ns.unified_train_height = 448

    model_path = Path(ns.pretrained_model_name_or_path)
    if not model_path.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_path}\n"
            "Set FLUX2_MODEL_DIR or pass --pretrained_model_name_or_path."
        )

    ns.run_root = Path(ns.run_root).resolve()
    ns.run_root.mkdir(parents=True, exist_ok=True)
    tb_dir = ns.run_root / "tb"
    proj_dir = ns.run_root / "accelerate_project"
    tb_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    random.seed(ns.seed)
    torch.manual_seed(ns.seed)

    tb_ok = _tensorboard_available()
    if not tb_ok:
        print(
            "\n*** `tensorboard` is not installed — Accelerate cannot write TensorBoard logs.\n"
            "    Fix: pip install tensorboard\n"
            "    Meanwhile: scalars go to validation_out/validation_metrics.jsonl; PNGs still save as usual.\n"
        )
    else:
        print(
            "TensorBoard package found.\n"
            "  Event files are created under: <run_root>/tb/validation_loop_smoke/ (not the tb/ root).\n"
            "  Use: tensorboard --logdir <run_root>/tb\n"
        )

    mixed = "bf16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        mixed_precision=mixed,
        log_with="tensorboard" if tb_ok else None,
        project_config=ProjectConfiguration(project_dir=str(proj_dir), logging_dir=str(tb_dir)),
    )
    device = accelerator.device
    weight_dtype = torch.float32
    if mixed == "fp16":
        weight_dtype = torch.float16
    elif mixed == "bf16":
        weight_dtype = torch.bfloat16

    args = _build_canvas_args(ns)

    print("Loading tokenizer / text encoder …")
    tokenizer = PixtralProcessor.from_pretrained(str(model_path / "tokenizer"))
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(str(model_path), subfolder="text_encoder")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_path), subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    print("Loading VAE …")
    vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae")
    print("Loading transformer (large) …")
    transformer_base = Flux2Transformer2DModelBase.from_pretrained(str(model_path), subfolder="transformer")
    transformer = FluxTransformer2DModel.from_config(transformer_base.config)
    transformer.load_state_dict(transformer_base.state_dict(), strict=True)
    del transformer_base

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)

    print("Attaching LoRA processors …")
    _attach_lora(transformer, args, device, weight_dtype)
    print(f"Randomizing LoRA weights (std={ns.dummy_lora_std}) …")
    _randomize_lora_weights(transformer, std=ns.dummy_lora_std)

    print("Building canvas dataset …")
    train_dataset = make_canvas_train_dataset(args, accelerator=None)
    print(f"  dataset len = {len(train_dataset)}")

    tb_run_dir = tb_dir / "validation_loop_smoke"
    accelerator.init_trackers("validation_loop_smoke", config={"dummy_lora": True, "global_step": ns.global_step})
    if tb_ok:
        accelerator.log({"smoke/models_ready": 1}, step=0)
        ev = list(tb_run_dir.glob("events.out.tfevents.*"))
        print(f"TensorBoard run directory: {tb_run_dir}")
        print(f"  event file(s) after init: {len(ev)} (expect ≥1 before validation finishes)")
        if ev:
            print(f"  example: {ev[0].name}")

    print(
        "\nRunning run_dataloader_validation — 1× full loss forward, then N × inference_steps DiT denoising.\n"
        "  This phase can take **several minutes** at 1920×1088; use --smoke_fast for a quick check.\n"
    )
    run_dataloader_validation(
        accelerator=accelerator,
        args=args,
        train_dataset=train_dataset,
        collate_fn=collate_fn_canvas,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler_copy=noise_scheduler_copy,
        weight_dtype=weight_dtype,
        global_step=ns.global_step,
    )

    step_dir = os.path.join(args.output_dir, args.validation_samples_subdir, f"step_{ns.global_step:07d}")
    print()
    print("Done.")
    print(f"  PNGs + prompts:     {step_dir}")
    print(f"  Metrics (JSONL):    {os.path.join(args.output_dir, 'validation_metrics.jsonl')}")
    if tb_ok:
        print(f"  TensorBoard logdir: {tb_dir}  (events under {tb_run_dir})")
        print(f"  Example:            tensorboard --logdir {tb_dir}")

    accelerator.end_training()


def test_validation_loop_smoke_env_gated(monkeypatch):
    """Optional pytest entry: ``RUN_FLUX2_VALIDATION_SMOKE=1 pytest ...::test_validation_loop_smoke_env_gated``."""
    import pytest

    if os.environ.get("RUN_FLUX2_VALIDATION_SMOKE") != "1":
        pytest.skip("Set RUN_FLUX2_VALIDATION_SMOKE=1 (needs FLUX.2 weights, CSV data, GPU recommended).")
    monkeypatch.chdir(_TRAIN_DIR)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_validation_loop_smoke",
            "--validation_inference_steps",
            "4",
            "--validation_num_samples",
            "1",
        ],
    )
    main()


if __name__ == "__main__":
    main()
