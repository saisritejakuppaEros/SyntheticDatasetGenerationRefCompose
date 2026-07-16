#!/usr/bin/env python3
"""
Dataloader smoke test (train.py canvas path).

With ``canvas_conditioning=bbox_multiview`` and **unified** resolution (no
``--canvas_random_target_resolution``):

  1) Original scene image from disk (``canvas_target_column`` / ``image_path``).
  2) Canvas: composite in ``canvas_bbox_compose`` at **snapped**
     ``unified_train_width``×``unified_train_height`` (e.g. **1280×720** for 720p), letterboxed
     full frame + bbox crops, then the **same** unified resize as the target
     (``canvas_unified_resize``) → ``subject_pixel_values`` matching ``pixel_values`` spatial size.
  3) Optional train-time canvas augments (bbox margin, brightness, perspective/shear on up to
     ``canvas_geom_extreme_max_crops`` crops, mild shear on the rest) when ``--canvas_augment``.
  4) Prompt: CSV ``prompt`` if present, else derived from ``class_name``s.

``--cond_size`` defaults to 512 for parity with ``train.py`` but does **not** set H×W here:
unified mode uses ``unified_train_width`` / ``unified_train_height`` (and ``load_image_safely``
only uses ``cond_size`` for a fallback placeholder on load failure).

Training adds noise only to **main** latents in ``train.py`` (flow-match); canvas latents are clean conditioning.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image
from torchvision.utils import save_image

# train.py cwd when launched via train_flux2_lora.sh
_TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from src.canvas_bbox_compose import build_canvas_for_image_rel
from src.canvas_dataset import make_canvas_train_dataset, collate_fn_canvas
from src.flux2_train_helpers import encode_flux2_latents
from src.jsonl_datasets import multiple_16, resolve_image_path

# Defaults from train_flux2_lora.sh (used directly)
DEFAULT_MODEL_DIR = (
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/"
    "snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
)
DEFAULT_CSV_PATH = (
    "/mnt/data0/teja/research_multiref/dataset_preparation/output/bbox_results/yolo26_detections.csv"
)
DEFAULT_CANVAS_IMAGE_ROOT = "/mnt/data0/teja/research_multiref/dataset_preparation/output/images"
DEFAULT_MULTIVIEW_DIR = "/mnt/data0/teja/research_multiref/dataset_preparation/output/multiview_out"


def _shape_str(x) -> str:
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    if isinstance(x, list):
        return f"list[len={len(x)}] (first elem type={type(x[0]).__name__})"
    return str(type(x).__name__)


def print_batch_shapes(batch: dict, prefix: str = "") -> None:
    for k, v in batch.items():
        print(f"{prefix}{k}: {_shape_str(v)}")
        if k == "prompts" and isinstance(v, list) and v:
            preview = v[0][:120] + ("…" if len(v[0]) > 120 else "")
            print(f"{prefix}  └─ prompts[0] preview: {preview!r}")


def save_normalized_chw(t: torch.Tensor, path: Path) -> None:
    """t: CHW in roughly [-1, 1] (dataset normalize)."""
    x = (t.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    save_image(x, path)


def _row_canvas_key(row: dict, canvas_column: str) -> str | None:
    key = canvas_column if canvas_column in row else "subject_path"
    if key in row and row[key]:
        return key
    return None


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Test canvas dataloader; print shapes and save sample tensors as PNG.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--canvas_image_root",
        type=str,
        default=DEFAULT_CANVAS_IMAGE_ROOT,
        help="Prepended to relative image_path / canvas_path values (matches train_flux2_lora.sh).",
    )
    parser.add_argument(
        "--canvas_conditioning",
        type=str,
        default="bbox_multiview",
        choices=["precomputed", "bbox_multiview"],
        help="Match train.py --canvas_conditioning.",
    )
    parser.add_argument("--canvas_bbox_min_side", type=float, default=0.0)
    parser.add_argument("--canvas_multiview_match_min_side", type=float, default=200.0)
    parser.add_argument("--canvas_multiview_dir", type=str, default=DEFAULT_MULTIVIEW_DIR)
    parser.add_argument("--canvas_multiview_prob", type=float, default=0.5)
    parser.add_argument("--canvas_background", type=str, default="black", choices=["black", "scaled"])
    parser.add_argument(
        "--canvas_geom_extreme_max_crops",
        type=int,
        default=3,
        help="Match train.py: max crops per image with extreme perspective/shear.",
    )
    parser.add_argument("--canvas_column", type=str, default="canvas_path")
    parser.add_argument("--canvas_target_column", type=str, default="image_path")
    parser.add_argument("--canvas_prompt_column", type=str, default="prompt")
    parser.add_argument(
        "--cond_size",
        type=int,
        default=512,
        help="Matches train.py (fallback placeholder in load_image_safely; random-res canvas subject transform). "
        "Ignored for unified target/canvas pixel H×W.",
    )
    parser.add_argument("--noise_size", type=int, default=1024)
    parser.add_argument("--unified_train_width", type=int, default=1920)
    parser.add_argument("--unified_train_height", type=int, default=1080)
    parser.add_argument(
        "--canvas_unified_resize",
        type=str,
        default="contain",
        choices=["cover", "contain"],
        help="Unified target/canvas box: contain (default) = letterbox; cover = scale+center-crop.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Fix RNG for more stable shapes across runs.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Flux2 checkpoint dir; VAE loaded from subfolder 'vae' for latent shapes (unless --skip_vae).",
    )
    parser.add_argument(
        "--skip_vae",
        action="store_true",
        help="Do not load VAE; skip latent shape prints and latent encoding.",
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        default=0,
        help="Sample index: CSV row (precomputed) or scene group index (bbox_multiview). -1 = random shuffle.",
    )
    parser.add_argument(
        "--canvas_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match train.py: bbox margin / brightness / shear on composed canvas.",
    )
    parser.add_argument(
        "--canvas_augment_bbox_margin_prob",
        type=float,
        default=1.0,
        help="Default 1.0 for this test script so bbox margin runs on every box (when canvas_augment).",
    )
    parser.add_argument("--canvas_augment_bbox_margin_min", type=int, default=10)
    parser.add_argument("--canvas_augment_bbox_margin_max", type=int, default=30)
    parser.add_argument("--canvas_augment_bbox_expand_prob", type=float, default=0.5)
    parser.add_argument(
        "--canvas_augment_brightness_prob",
        type=float,
        default=1.0,
        help="Default 1.0 here so brightness runs every crop for testing; use e.g. 0.88 to match train.py.",
    )
    parser.add_argument(
        "--canvas_augment_brightness_bimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match train.py: bimodal dim vs lit bands (use --no-canvas_augment_brightness_bimodal for uniform).",
    )
    parser.add_argument("--canvas_augment_brightness_min", type=float, default=0.45)
    parser.add_argument("--canvas_augment_brightness_max", type=float, default=1.85)
    parser.add_argument("--canvas_augment_brightness_dim_min", type=float, default=0.32)
    parser.add_argument("--canvas_augment_brightness_dim_max", type=float, default=0.58)
    parser.add_argument("--canvas_augment_brightness_lit_min", type=float, default=1.42)
    parser.add_argument("--canvas_augment_brightness_lit_max", type=float, default=2.08)
    parser.add_argument(
        "--canvas_augment_global_brightness_prob",
        type=float,
        default=1.0,
        help="Default 1.0 here so global scene brightness always runs when canvas_augment; use e.g. 0.88 for train.py.",
    )
    parser.add_argument(
        "--canvas_augment_perspective_prob",
        type=float,
        default=1.0,
        help="Default 1.0 here so perspective runs every crop when testing; use e.g. 0.48 for train.py.",
    )
    parser.add_argument(
        "--canvas_augment_perspective_distortion",
        type=float,
        default=0.62,
        help="Match train.py: torchvision-style perspective strength.",
    )
    parser.add_argument(
        "--canvas_augment_shear_prob",
        type=float,
        default=1.0,
        help="Default 1.0 here so shear runs every crop for testing; use e.g. 0.55 for train.py.",
    )
    parser.add_argument("--canvas_augment_shear_degrees", type=float, default=26.0)
    args = parser.parse_args()

    random_res = getattr(args, "canvas_random_target_resolution", False)
    if args.train_batch_size != 1 and random_res:
        print(
            "Warning: --canvas_random_target_resolution uses variable target H×W; collate needs equal "
            f"sizes per batch. Forcing train_batch_size=1 (was {args.train_batch_size}).",
            file=sys.stderr,
        )
        args.train_batch_size = 1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    args.dataset_type = "canvas"
    # train.py sets spatial_column; canvas path does not use it
    args.spatial_column = "None"

    print("=== Dataloader test (canvas) ===")
    print(f"Output images directory: {out_dir}")
    print(f"CSV_PATH (input): {args.csv_path}")
    print(f"canvas_image_root: {args.canvas_image_root}")
    print(f"canvas_conditioning: {args.canvas_conditioning}")
    uw, uh = multiple_16(args.unified_train_width), multiple_16(args.unified_train_height)
    print(
        f"cond_size={args.cond_size}, noise_size={args.noise_size}, "
        f"unified(nominal)={args.unified_train_width}×{args.unified_train_height} → snapped={uw}×{uh}, "
        f"canvas_unified_resize={args.canvas_unified_resize}, batch_size={args.train_batch_size}, "
        f"canvas_augment={args.canvas_augment}"
    )
    print()

    train_dataset = make_canvas_train_dataset(args, accelerator=None)
    print(f"Dataset length: {len(train_dataset)}")

    use_row_for_files = args.dataset_index >= 0
    if use_row_for_files:
        if not (0 <= args.dataset_index < len(train_dataset)):
            raise SystemExit(f"dataset_index {args.dataset_index} out of range [0, {len(train_dataset)})")
        batch = collate_fn_canvas([train_dataset[args.dataset_index]])
        print(f"Batch from dataset index {args.dataset_index} (matches saved originals).")
        if train_dataset.conditioning == "bbox_multiview":
            rel, _gr = train_dataset.groups[args.dataset_index]
            row = {args.canvas_target_column: rel}
        else:
            row = train_dataset.rows[args.dataset_index]
    else:
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn_canvas,
            num_workers=args.dataloader_num_workers,
        )
        batch = next(iter(loader))
        print("Batch from first shuffled DataLoader step (original/canvas PNGs from CSV not saved).")
        row = None

    pv = batch["pixel_values"]
    _, _, th, tw = pv.shape
    print()
    print("--- Three training inputs (same as train.py) ---")
    print(f"1) prompt: {batch['prompts'][0]!r}")
    print(f"2) target — processed tensor spatial size: {th}×{tw} (batch shape {tuple(pv.shape)})")
    if "subject_pixel_values" in batch:
        _, _, ch, cw = batch["subject_pixel_values"].shape
        print(
            f"3) canvas  — processed tensor spatial size: {ch}×{cw} "
            f"(unified box, same as target; batch shape {tuple(batch['subject_pixel_values'].shape)})"
        )
    else:
        print("3) canvas  — absent in this row (no non-empty canvas_path / subject_path in CSV)")

    print()
    print("--- Batch tensor / list shapes ---")
    print_batch_shapes(batch)
    print()

    if not args.skip_vae:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        print(f"Loading VAE from {args.pretrained_model_name_or_path} (subfolder=vae), device={device}, dtype={weight_dtype}")
        vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
        vae.eval().requires_grad_(False)
        vae.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            main_latent = encode_flux2_latents(
                vae, batch["pixel_values"].to(device=device, dtype=torch.float32), weight_dtype
            )
        print(f"Target latent after encode_flux2_latents (main branch): {tuple(main_latent.shape)}")
        if "subject_pixel_values" in batch:
            with torch.no_grad():
                sub_latent = encode_flux2_latents(
                    vae,
                    batch["subject_pixel_values"].to(device=device, dtype=torch.float32),
                    weight_dtype,
                )
            print(f"Canvas latent after encode_flux2_latents (subject branch): {tuple(sub_latent.shape)}")
        print()

    root = args.canvas_image_root.strip() or None
    if use_row_for_files and row is not None:
        tgt_rel = row.get(args.canvas_target_column, "")
        tgt_abs = resolve_image_path(tgt_rel, root)
        if tgt_rel:
            img = Image.open(tgt_abs).convert("RGB")
            w0, h0 = img.size
            print(f"Original target file: {tgt_abs} — size {w0}×{h0} (W×H)")
            img.save(out_dir / "original_target_image.png")
            print(f"Saved {out_dir / 'original_target_image.png'}")

        if train_dataset.conditioning == "bbox_multiview":
            rel_g, group_rows = train_dataset.groups[args.dataset_index]
            full_pil = Image.open(resolve_image_path(rel_g, root)).convert("RGB")
            cw, ch = train_dataset._compose_canvas_wh
            aug_kw = train_dataset._canvas_augment_kwargs()
            comp_clean = build_canvas_for_image_rel(
                rel_g,
                full_pil,
                group_rows,
                train_dataset._multiview_root,
                train_dataset._multiview_prob,
                train_dataset._canvas_background,
                args.seed,
                train_dataset._bbox_min_side,
                cw,
                ch,
                extreme_geom_max_crops=args.canvas_geom_extreme_max_crops,
                augment=False,
            )
            out_clean = out_dir / f"composed_canvas_{cw}x{ch}_no_aug.png"
            comp_clean.save(out_clean)
            print(
                f"Composed canvas (no aug, pre-unified resize): {comp_clean.size[0]}×{comp_clean.size[1]} (W×H) → {out_clean}"
            )
            comp_aug = build_canvas_for_image_rel(
                rel_g,
                full_pil,
                group_rows,
                train_dataset._multiview_root,
                train_dataset._multiview_prob,
                train_dataset._canvas_background,
                args.seed,
                train_dataset._bbox_min_side,
                cw,
                ch,
                extreme_geom_max_crops=args.canvas_geom_extreme_max_crops,
                **aug_kw,
            )
            out_aug = out_dir / f"composed_canvas_{cw}x{ch}_aug.png"
            comp_aug.save(out_aug)
            print(f"Composed canvas (with aug): {comp_aug.size[0]}×{comp_aug.size[1]} (W×H) → {out_aug}")
        else:
            ck = _row_canvas_key(row, args.canvas_column)
            if ck:
                c_rel = row[ck]
                c_abs = resolve_image_path(c_rel, root)
                cimg = Image.open(c_abs).convert("RGB")
                cw0, ch0 = cimg.size
                print(f"Original canvas file: {c_abs} — size {cw0}×{ch0} (W×H)")
                cimg.save(out_dir / "original_canvas_image.png")
                print(f"Saved {out_dir / 'original_canvas_image.png'}")

    # Processed tensors (what the model sees as pixels before VAE)
    b = batch["pixel_values"].shape[0]
    for i in range(b):
        save_normalized_chw(batch["pixel_values"][i], out_dir / f"processed_target_b{i}.png")
        print(f"Saved {out_dir / f'processed_target_b{i}.png'} (normalized preview, spatial {th}×{tw})")
    if "subject_pixel_values" in batch:
        _, _, ch, cw = batch["subject_pixel_values"].shape
        for i in range(b):
            save_normalized_chw(
                batch["subject_pixel_values"][i],
                out_dir / f"processed_canvas_cond_{ch}x{cw}_b{i}.png",
            )
            print(f"Saved {out_dir / f'processed_canvas_cond_{ch}x{cw}_b{i}.png'} (canvas after unified resize, same H×W as target)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
