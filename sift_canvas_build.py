#!/usr/bin/env python3
"""
Build a conditioning canvas by warping reference images onto the layout of a
generated multi-reference image.

Uses LoMa (external/LoMa) for feature matching between each reference and the
generated image, estimates a homography (with similarity / bbox fallbacks), and
composites layers with landscapes/backgrounds first and foreground subjects on top.

Example (project venv):
    .venv/bin/python sift_canvas_build.py \\
        --metadata outputs/stage3_generated/metadata/sample_0000008.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
LOMA_SRC = SCRIPT_DIR / "external" / "LoMa" / "src"
if str(LOMA_SRC) not in sys.path:
    sys.path.insert(0, str(LOMA_SRC))

from loma import LoMa, LoMaB  # noqa: E402


BACKGROUND_MARKERS = ("/landscapes/", "/landscape/", "/backgrounds/", "/background/")


@dataclass
class ReferencePlacement:
    path: Path
    is_background: bool
    homography: np.ndarray | None
    num_matches: int
    num_inliers: int
    method: str
    coverage: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reference canvas via LoMa feature matching.")
    p.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to stage3 metadata JSON (e.g. outputs/stage3_generated/metadata/sample_0000008.json).",
    )
    p.add_argument(
        "--images_root",
        type=str,
        default=str(SCRIPT_DIR),
        help="Root for reference_images paths in metadata (default: synthetic_dataset_generation/).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where to write canvas outputs (default: <metadata_dir>/../canvas).",
    )
    p.add_argument(
        "--min_matches",
        type=int,
        default=4,
        help="Minimum LoMa matches before attempting a transform.",
    )
    p.add_argument(
        "--min_inliers",
        type=int,
        default=4,
        help="Minimum RANSAC inliers to accept a homography.",
    )
    p.add_argument(
        "--save_debug",
        action="store_true",
        help="Also save a side-by-side debug panel (generated | canvas | overlay).",
    )
    p.add_argument(
        "--save_matches",
        action="store_true",
        help="Also save per-reference match visualizations.",
    )
    return p.parse_args()


def resolve_path(raw: str, root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def is_background_reference(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(marker in s for marker in BACKGROUND_MARKERS)


def load_bgr(path: Path, max_side: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if max_side is not None:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(
                img,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    return img


def estimate_transform(
    kpts_ref: np.ndarray,
    kpts_gen: np.ndarray,
    *,
    min_inliers: int,
) -> tuple[np.ndarray | None, int, str]:
    """Map reference keypoints -> generated-image keypoints."""
    if len(kpts_ref) < 4:
        return None, 0, "insufficient_matches"

    H, mask = cv2.findHomography(
        kpts_ref,
        kpts_gen,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.995,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    if H is not None and inliers >= min_inliers:
        return H, inliers, "homography"

    M, inl_mask = cv2.estimateAffinePartial2D(
        kpts_ref,
        kpts_gen,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.995,
    )
    inl = int(inl_mask.sum()) if inl_mask is not None else 0
    if M is not None and inl >= min_inliers:
        H2 = np.vstack([M, [0.0, 0.0, 1.0]])
        return H2, inl, "similarity"

    ref_c = kpts_ref.mean(axis=0)
    gen_c = kpts_gen.mean(axis=0)
    ref_span = max(float(np.ptp(kpts_ref[:, 0])), float(np.ptp(kpts_ref[:, 1])), 1.0)
    gen_span = max(float(np.ptp(kpts_gen[:, 0])), float(np.ptp(kpts_gen[:, 1])), 1.0)
    scale = gen_span / ref_span
    tx = gen_c[0] - scale * ref_c[0]
    ty = gen_c[1] - scale * ref_c[1]
    H3 = np.array(
        [[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return H3, len(kpts_ref), "bbox_center_scale"


def warp_reference(
    ref_bgr: np.ndarray,
    H: np.ndarray,
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    warped = cv2.warpPerspective(
        ref_bgr,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = np.zeros((ref_bgr.shape[0], ref_bgr.shape[1]), dtype=np.uint8)
    mask[:, :] = 255
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, warped_mask


def composite_layer(canvas: np.ndarray, layer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = canvas.astype(np.float32) * (1.0 - alpha) + layer.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def match_coverage(kpts_gen: np.ndarray, out_w: int, out_h: int) -> float:
    if len(kpts_gen) == 0:
        return 0.0
    x1, y1 = kpts_gen.min(axis=0)
    x2, y2 = kpts_gen.max(axis=0)
    box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    return box_area / float(out_w * out_h)


def sort_references(placements: list[ReferencePlacement]) -> list[ReferencePlacement]:
    """Backgrounds first (larger coverage first), then foregrounds (larger coverage first)."""
    backgrounds = [p for p in placements if p.is_background]
    foregrounds = [p for p in placements if not p.is_background]
    backgrounds.sort(key=lambda p: p.coverage, reverse=True)
    foregrounds.sort(key=lambda p: p.coverage, reverse=True)
    return backgrounds + foregrounds


def draw_matches_panel(
    ref_path: Path,
    gen_path: Path,
    kpts_ref: np.ndarray,
    kpts_gen: np.ndarray,
) -> Image.Image:
    ref = Image.open(ref_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")
    w1, h1 = ref.size
    w2, h2 = gen.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), (0, 0, 0))
    canvas.paste(ref, (0, 0))
    canvas.paste(gen, (w1, 0))
    draw = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(0)
    for (x1, y1), (x2, y2) in zip(kpts_ref, kpts_gen):
        color = tuple(int(c) for c in rng.integers(40, 220, 3))
        draw.line([(x1, y1), (x2 + w1, y2)], fill=color, width=1)
    return canvas


def build_canvas_for_sample(
    metadata: dict[str, Any],
    *,
    images_root: Path,
    model: LoMa,
    min_matches: int,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, list[ReferencePlacement], dict[str, Any]]:
    gen_path = resolve_path(metadata["output_image"], images_root)
    gen_bgr = load_bgr(gen_path)
    out_h, out_w = gen_bgr.shape[:2]

    if "gen_width" in metadata and "gen_height" in metadata:
        target_w = int(metadata["gen_width"])
        target_h = int(metadata["gen_height"])
        if (out_w, out_h) != (target_w, target_h):
            gen_bgr = cv2.resize(gen_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out_w, out_h = target_w, target_h

    placements: list[ReferencePlacement] = []
    debug_matches: dict[str, Any] = {}

    for rel in metadata["reference_images"]:
        ref_path = resolve_path(rel, images_root)
        kpts_ref, kpts_gen = model.match(str(ref_path), str(gen_path))

        H, inliers, method = estimate_transform(
            kpts_ref,
            kpts_gen,
            min_inliers=min_inliers,
        )
        coverage = match_coverage(kpts_gen, out_w, out_h)
        bg = is_background_reference(ref_path)

        placements.append(
            ReferencePlacement(
                path=ref_path,
                is_background=bg,
                homography=H,
                num_matches=len(kpts_ref),
                num_inliers=inliers,
                method=method,
                coverage=coverage,
            )
        )
        debug_matches[str(ref_path)] = {
            "kpts_ref": kpts_ref,
            "kpts_gen": kpts_gen,
            "is_background": bg,
        }

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ordered = sort_references(placements)

    for placement in ordered:
        if placement.homography is None or placement.num_matches < min_matches:
            continue
        ref_bgr = load_bgr(placement.path, max_side=2048)
        warped, mask = warp_reference(ref_bgr, placement.homography, out_w, out_h)
        canvas = composite_layer(canvas, warped, mask)

    return canvas, gen_bgr, ordered, debug_matches


def save_debug_panel(
    gen_bgr: np.ndarray,
    canvas_bgr: np.ndarray,
    out_path: Path,
) -> None:
    overlay = cv2.addWeighted(gen_bgr, 0.45, canvas_bgr, 0.55, 0)
    panel = np.hstack([gen_bgr, canvas_bgr, overlay])
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    args = parse_args()
    meta_path = Path(args.metadata).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = meta_path.parent.parent / "canvas"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_id = metadata.get("id", meta_path.stem)
    print(f"Loading LoMa matcher...")
    model = LoMa(LoMaB())

    print(f"Building canvas for {sample_id}...")
    canvas_bgr, gen_bgr, placements, debug_matches = build_canvas_for_sample(
        metadata,
        images_root=images_root,
        model=model,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
    )

    canvas_path = output_dir / f"{sample_id}_canvas.png"
    cv2.imwrite(str(canvas_path), canvas_bgr)
    print(f"Saved canvas: {canvas_path}")

    meta_out = {
        "id": sample_id,
        "source_metadata": str(meta_path),
        "canvas_image": str(canvas_path),
        "generated_image": metadata.get("output_image"),
        "placements": [
            {
                "reference": str(p.path),
                "is_background": p.is_background,
                "num_matches": p.num_matches,
                "num_inliers": p.num_inliers,
                "method": p.method,
                "coverage": round(p.coverage, 4),
            }
            for p in placements
        ],
    }
    meta_out_path = output_dir / f"{sample_id}_canvas.json"
    meta_out_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_out_path}")

    if args.save_debug:
        debug_path = output_dir / f"{sample_id}_debug.png"
        save_debug_panel(gen_bgr, canvas_bgr, debug_path)
        print(f"Saved debug panel: {debug_path}")

    if args.save_matches:
        gen_path = resolve_path(metadata["output_image"], images_root)
        for ref_path_str, info in debug_matches.items():
            ref_path = Path(ref_path_str)
            stem = ref_path.stem
            match_path = output_dir / f"{sample_id}_matches_{stem}.jpg"
            panel = draw_matches_panel(
                ref_path,
                gen_path,
                info["kpts_ref"],
                info["kpts_gen"],
            )
            panel.save(match_path, quality=92)
            print(f"Saved matches: {match_path}")


if __name__ == "__main__":
    main()
