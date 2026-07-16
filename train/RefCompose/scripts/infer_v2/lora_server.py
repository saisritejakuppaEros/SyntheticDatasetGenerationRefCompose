#!/usr/bin/env python3
"""
FastAPI LoRA inference server for ``stage3_lora_infer.py``.

Loads FLUX.2 + canvas/depth LoRA once, then serves POST /generate requests.
Each request reads inputs from a folder on disk:

  - black_canvas.jpg
  - image_gen_depth.png
  - image_gen.txt

Writes ``image_gen_lora.png`` (+ ``image_gen_lora.txt`` sidecar) into the same folder.

Run::

    cd infer_v2
    CUDA_VISIBLE_DEVICES=0 python lora_server.py --port 8767
"""

from __future__ import annotations

import argparse
import copy
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

_INFER_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _INFER_DIR.parent / "train"
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
from src.prompt_helper import encode_prompts_flux2  # noqa: E402
from src.transformer_flux import FluxTransformer2DModel  # noqa: E402

import stage3_lora_infer as stage3  # noqa: E402

_infer_lock = threading.Lock()
_runtime: Optional["FluxLoraRuntime"] = None
_model_error: Optional[str] = None

CANVAS_NAME = "black_canvas.jpg"
DEPTH_NAME = "image_gen_depth.png"
PROMPT_NAME = "image_gen.txt"
OUTPUT_NAME = "image_gen_lora.png"
API_VERSION = 2


@dataclass
class FluxLoraRuntime:
    device: torch.device
    weight_dtype: torch.dtype
    tokenizer: Any
    text_encoder: Any
    noise_scheduler_copy: Any
    vae: Any
    transformer: FluxTransformer2DModel
    lora_w: list[float]
    unified_w: int
    unified_h: int
    max_sequence_length: int
    text_encoder_out_layers: tuple[int, ...]


class GenerateRequest(BaseModel):
    folder_path: str = Field(..., description="Folder with black_canvas.jpg, image_gen_depth.png, image_gen.txt")
    guidance_scale: float = Field(..., ge=0.0, description="CFG guidance scale for this run")
    seed: int = 42
    num_inference_steps: int = 28
    lora_first_depth_steps: int = 0
    lora_interval_appearance_strength: float = 1.0
    target_init: str = "black"
    canvas_depth_coarse_augment: bool = False
    no_canvas: bool = False
    no_depth: bool = False


class GenerateResponse(BaseModel):
    status: str
    folder_path: str
    guidance_scale: float
    output_path: str
    prompt_preview: str
    use_canvas: bool
    use_depth: bool
    canvas_source: str
    depth_source: str


def resolve_folder_inputs(
    folder: Path,
    *,
    no_canvas: bool = False,
    no_depth: bool = False,
) -> tuple[Optional[Path], Optional[Path], str]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")

    canvas_path = folder / CANVAS_NAME if not no_canvas else None
    depth_path = folder / DEPTH_NAME if not no_depth else None
    prompt_path = folder / PROMPT_NAME

    missing: list[str] = []
    if canvas_path is not None and not canvas_path.is_file():
        missing.append(CANVAS_NAME)
    if depth_path is not None and not depth_path.is_file():
        missing.append(DEPTH_NAME)
    if not prompt_path.is_file():
        missing.append(PROMPT_NAME)
    if missing:
        raise FileNotFoundError(
            f"folder {folder} is missing required file(s): {', '.join(missing)} "
            f"(expected {PROMPT_NAME}"
            f"{'' if no_canvas else f', {CANVAS_NAME}'}"
            f"{'' if no_depth else f', {DEPTH_NAME}'})"
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_path}")
    return canvas_path, depth_path, prompt


def _black_rgb(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (0, 0, 0))


def _resize_cover_single(im: Image.Image, tw: int, th: int) -> Image.Image:
    return stage3._resize_cover_aligned([im, im], tw, th, ref_index=0)[0]


def load_runtime(
    *,
    pretrained_model_name_or_path: str,
    lora_path: str,
    unified_width: int,
    unified_height: int,
    lora_num: int,
    ranks: list[int],
    network_alphas: list[int],
    max_sequence_length: int,
    text_encoder_out_layers: list[int],
) -> FluxLoraRuntime:
    lora_path_p = stage3.print_lora_checkpoint_before_load(lora_path, prefix="[lora_server]")
    if not lora_path_p.is_file():
        raise FileNotFoundError(f"lora_path not found: {lora_path_p}")
    model_path = Path(pretrained_model_name_or_path).expanduser().resolve()
    print(f"[lora_server] FLUX.2 base model: {model_path}", flush=True)
    if not model_path.is_dir():
        raise FileNotFoundError(f"pretrained_model_name_or_path not a directory: {model_path}")

    tw = multiple_16(unified_width)
    th = multiple_16(unified_height)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    rks = list(ranks)
    als = list(network_alphas)
    while len(rks) < lora_num:
        rks.append(rks[-1] if rks else 32)
    while len(als) < lora_num:
        als.append(als[-1] if als else 32)
    rks = rks[:lora_num]
    als = als[:lora_num]

    print(f"[lora_server] Loading FLUX.2 + LoRA once. device={device} unified={tw}x{th} ...", flush=True)
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

    lora_w = stage3._attach_lora(
        transformer,
        device=device,
        weight_dtype=weight_dtype,
        lora_num=lora_num,
        ranks=rks,
        network_alphas=als,
        unified_w=tw,
        unified_h=th,
    )

    lora_sd = load_file(str(lora_path_p))
    inc = transformer.load_state_dict(lora_sd, strict=False)
    unexpected = getattr(inc, "unexpected_keys", inc[1] if isinstance(inc, tuple) else [])
    print(f"[lora_server] Merged LoRA ({len(lora_sd)} tensors from {lora_path_p.name}).", flush=True)
    if unexpected:
        print(f"[lora_server] Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}", flush=True)

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    return FluxLoraRuntime(
        device=device,
        weight_dtype=weight_dtype,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        noise_scheduler_copy=noise_scheduler_copy,
        vae=vae,
        transformer=transformer,
        lora_w=lora_w,
        unified_w=tw,
        unified_h=th,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=tuple(text_encoder_out_layers),
    )


def _ns_like_for_augment(augment: bool) -> Any:
    class _NS:
        canvas_depth_coarse_augment = True
        canvas_depth_coarse_blur_prob = 0.7
        canvas_depth_coarse_noise_prob = 0.4
        canvas_depth_coarse_patch_prob = 0.3

    ns = _NS()
    ns.canvas_depth_coarse_augment = augment
    return ns


def _prepare_conditioning(
    rt: FluxLoraRuntime,
    req: GenerateRequest,
    canvas_path: Optional[Path],
    depth_path: Optional[Path],
) -> tuple[Image.Image, Image.Image, str, str]:
    """Build canvas/depth tensors. When disabled, returns black images and never reads the file."""
    tw, th = rt.unified_w, rt.unified_h
    use_canvas = not req.no_canvas
    use_depth = not req.no_depth

    if use_canvas:
        assert canvas_path is not None
        canvas_pil = Image.open(canvas_path).convert("RGB")
        canvas_source = f"file:{canvas_path.name}"
    else:
        canvas_pil = None
        canvas_source = "black"

    if use_depth:
        assert depth_path is not None
        depth_pil = load_depth_image_as_rgb_pil(depth_path)
        depth_source = f"file:{depth_path.name}"
    else:
        depth_pil = None
        depth_source = "black"

    if use_canvas and use_depth:
        canvas_u, depth_u = stage3._resize_cover_aligned(
            [canvas_pil, depth_pil], tw, th, ref_index=1
        )
    elif use_canvas:
        canvas_u = _resize_cover_single(canvas_pil, tw, th)
        depth_u = _black_rgb(tw, th)
    elif use_depth:
        depth_u = _resize_cover_single(depth_pil, tw, th)
        canvas_u = _black_rgb(tw, th)
    else:
        canvas_u = _black_rgb(tw, th)
        depth_u = _black_rgb(tw, th)

    if req.canvas_depth_coarse_augment and use_depth:
        depth_u = coarse_degrade_depth_rgb_pil(depth_u, _ns_like_for_augment(True))

    return canvas_u, depth_u, canvas_source, depth_source


def run_generate_from_folder(rt: FluxLoraRuntime, req: GenerateRequest) -> GenerateResponse:
    folder = Path(req.folder_path).expanduser().resolve()
    canvas_path, depth_path, prompt = resolve_folder_inputs(
        folder,
        no_canvas=req.no_canvas,
        no_depth=req.no_depth,
    )
    tw, th = rt.unified_w, rt.unified_h
    use_canvas = not req.no_canvas
    use_depth = not req.no_depth

    canvas_u, depth_u, canvas_source, depth_source = _prepare_conditioning(
        rt, req, canvas_path, depth_path
    )
    print(
        f"[lora_server] Conditioning: canvas={canvas_source}, depth={depth_source}",
        flush=True,
    )

    target_init = req.target_init
    if req.no_canvas and target_init == "canvas":
        target_init = "black"
        print("[lora_server] target_init forced to black because no_canvas is set", flush=True)

    if target_init == "canvas":
        target_u = canvas_u.copy()
    else:
        target_u = Image.new("RGB", (tw, th), (0, 0, 0))

    pv = pil_to_model_tensor(target_u).unsqueeze(0)
    subj_full = pil_to_model_tensor(canvas_u).unsqueeze(0)
    cond_full = pil_to_model_tensor(depth_u).unsqueeze(0)

    pe, tid = encode_prompts_flux2(
        rt.text_encoder,
        rt.tokenizer,
        [prompt],
        rt.device,
        rt.max_sequence_length,
        rt.weight_dtype,
        rt.text_encoder_out_layers,
    )

    lora_w = rt.lora_w
    lora_limited = int(req.lora_first_depth_steps) > 0

    def _lora_weighter(step_i: int, n_steps: int) -> None:
        ast = min(max(float(req.lora_interval_appearance_strength), 0.0), 1.5)
        n_depth = int(req.lora_first_depth_steps)
        if n_depth > 0:
            if step_i < n_depth:
                wc, wd = 0.2, 1.0
            else:
                wc, wd = 1.0, 0.1
            lora_w[0], lora_w[1] = wc * ast, wd

    gen = torch.Generator(device=rt.device).manual_seed(int(req.seed))
    with torch.inference_mode():
        out_pil = _denoise_one(
            vae=rt.vae,
            transformer=rt.transformer,
            scheduler_template=rt.noise_scheduler_copy,
            pixel_values=pv,
            subject_pixel_values=subj_full,
            cond_pixel_values=cond_full,
            prompt_embeds=pe,
            text_ids=tid,
            guidance_scale=float(req.guidance_scale),
            weight_dtype=rt.weight_dtype,
            device=rt.device,
            num_inference_steps=int(req.num_inference_steps),
            generator=gen,
            lora_weighter=_lora_weighter if lora_limited else None,
        )

    output_path = folder / OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_pil.save(output_path)
    output_path.with_suffix(".txt").write_text(prompt, encoding="utf-8")

    preview = prompt[:120] + ("…" if len(prompt) > 120 else "")
    cond_note = []
    if req.no_canvas:
        cond_note.append("no_canvas")
    if req.no_depth:
        cond_note.append("no_depth")
    cond_str = f" ({', '.join(cond_note)})" if cond_note else ""
    print(
        f"[lora_server] Saved guidance_scale={req.guidance_scale:g}{cond_str} -> {output_path}",
        flush=True,
    )
    return GenerateResponse(
        status="ok",
        folder_path=str(folder),
        guidance_scale=float(req.guidance_scale),
        output_path=str(output_path),
        prompt_preview=preview,
        use_canvas=use_canvas,
        use_depth=use_depth,
        canvas_source=canvas_source,
        depth_source=depth_source,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastAPI LoRA inference server (stage3)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=stage3._DEFAULT_FLUX2)
    p.add_argument("--lora_path", type=str, default=stage3._DEFAULT_LORA)
    p.add_argument("--unified_width", type=int, default=stage3.UNIFIED_WIDTH)
    p.add_argument("--unified_height", type=int, default=stage3.UNIFIED_HEIGHT)
    p.add_argument("--lora_num", type=int, default=2)
    p.add_argument("--ranks", type=int, nargs="+", default=[32])
    p.add_argument("--network_alphas", type=int, nargs="+", default=[32])
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=[10, 20, 30])
    return p


def create_app(cfg: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _runtime, _model_error
        _model_error = None
        _runtime = None
        try:
            _runtime = load_runtime(
                pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
                lora_path=cfg.lora_path,
                unified_width=cfg.unified_width,
                unified_height=cfg.unified_height,
                lora_num=cfg.lora_num,
                ranks=list(cfg.ranks),
                network_alphas=list(cfg.network_alphas),
                max_sequence_length=cfg.max_sequence_length,
                text_encoder_out_layers=list(cfg.text_encoder_out_layers),
            )
            print("[lora_server] Ready.", flush=True)
        except Exception as e:
            _model_error = f"{type(e).__name__}: {e}"
            print(f"[lora_server] FATAL: {_model_error}", flush=True)
        yield
        _runtime = None

    app = FastAPI(title="Stage3 FLUX+LoRA Server", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ok" if _runtime is not None else "error",
            "api_version": API_VERSION,
            "error": _model_error,
            "device": str(_runtime.device) if _runtime else None,
            "unified_resolution": f"{_runtime.unified_w}x{_runtime.unified_h}" if _runtime else None,
            "expected_inputs": [CANVAS_NAME, DEPTH_NAME, PROMPT_NAME],
            "optional_flags": ["no_canvas", "no_depth"],
            "output_name": OUTPUT_NAME,
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        if _runtime is None:
            raise HTTPException(
                status_code=503,
                detail=_model_error or "Server is starting up or failed to load models.",
            )
        if req.target_init not in {"black", "canvas"}:
            raise HTTPException(status_code=400, detail="target_init must be 'black' or 'canvas'")

        with _infer_lock:
            try:
                return run_generate_from_folder(_runtime, req)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except torch.cuda.OutOfMemoryError as e:
                raise HTTPException(status_code=507, detail=f"CUDA OOM: {e}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference failed: {type(e).__name__}: {e}",
                ) from e

    return app


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
