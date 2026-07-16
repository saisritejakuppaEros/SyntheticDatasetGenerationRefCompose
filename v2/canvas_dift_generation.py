#!/usr/bin/env python3
"""Build DIFT-rotated canvases for synthetic_dataset_generation/v2.

For each sample:
  - theme GT:   outputs/theme_images/images/<id>.png
  - refs:       outputs/reference_images/<id>/bbox{N}.jpg
  - place each ref into its Grounding-DINO source_bbox (bbox-anchored scale)
  - optionally tweak rotation via DIFT matches restricted *inside* that bbox
  - accept DIFT rotation only if RANSAC inliers are strong and estimated scale
    agrees with the bbox scale within a relative tolerance (default 30%)

Output:
  outputs/canvas_dift/images/<id>.png
  outputs/canvas_dift/metadata/<id>.json

Example:
  conda run -n pixart_parth python canvas_dift_generation.py --limit 4 --gpu_id 2
  GPU_ID=3 bash run_canvas_dift.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_CODES = Path("/mnt/data0/teja/research_multiref/parth_pipeline/codes")
if str(PIPELINE_CODES) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODES))

from dift_backend import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    DEFAULT_IMG_SIZE,
    DEFAULT_NUM_QUERIES,
    DEFAULT_T,
    DEFAULT_UP_FT_INDEX,
    get_featurizer,
    match_ref_to_gt_correspondences,
)

DEFAULT_THEME_DIR = SCRIPT_DIR / "outputs/theme_images/images"
DEFAULT_REFERENCE_DIR = SCRIPT_DIR / "outputs/reference_images"
DEFAULT_BBOX_META_DIR = SCRIPT_DIR / "outputs/bbox_annotations/metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/canvas_dift"
DEFAULT_WHITE_THRESHOLD = 240
DEFAULT_SCALE_TOL = 0.30
DEFAULT_RANSAC_THRESH = 8.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bbox-anchored canvas with optional DIFT rotation tweak."
    )
    p.add_argument("--theme_dir", type=Path, default=DEFAULT_THEME_DIR)
    p.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    p.add_argument("--bbox_metadata_dir", type=Path, default=DEFAULT_BBOX_META_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--theme", type=str, default=None, help="Process only this sample id")
    p.add_argument("--limit", type=int, default=0, help="Process at most N samples (0=all)")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--gpu_id", type=str, default=os.environ.get("GPU_ID", "0"))
    p.add_argument("--white_threshold", type=int, default=DEFAULT_WHITE_THRESHOLD)
    p.add_argument("--min_matches", type=int, default=12)
    p.add_argument("--min_inliers", type=int, default=12)
    p.add_argument("--ransac_thresh", type=float, default=DEFAULT_RANSAC_THRESH)
    p.add_argument(
        "--scale_tol",
        type=float,
        default=DEFAULT_SCALE_TOL,
        help="Reject DIFT rotation if |s_dift/s_bbox - 1| exceeds this.",
    )
    p.add_argument(
        "--fit",
        choices=("contain", "cover"),
        default="contain",
        help="How ref aspect fits into source_bbox before optional rotation.",
    )
    p.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--ensemble", type=int, default=DEFAULT_ENSEMBLE)
    p.add_argument("--canvas_bg", choices=("white", "black"), default="white")
    p.add_argument(
        "--sd_model",
        type=str,
        default=os.environ.get(
            "DIFT_SD_MODEL",
            "/tmp/sd15_dift_local",
        ),
        help="Local SD checkpoint path or Hub id for DIFT features.",
    )
    p.add_argument("--save_debug", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_sample_ids(theme_dir: Path, reference_dir: Path, theme: str | None) -> list[str]:
    if theme:
        return [theme]
    ids: list[str] = []
    for p in sorted(theme_dir.glob("*.png")):
        sid = p.stem
        ref_dir = reference_dir / sid
        if ref_dir.is_dir() and any(ref_dir.glob("bbox*.jpg")):
            ids.append(sid)
    return ids


def collect_refs(sample_id: str, reference_dir: Path, bbox_meta_dir: Path) -> list[dict[str, Any]]:
    sample_ref_dir = reference_dir / sample_id
    refs: list[dict[str, Any]] = []

    bbox_meta_path = bbox_meta_dir / f"{sample_id}.json"
    objects = []
    if bbox_meta_path.is_file():
        meta = load_json(bbox_meta_path)
        objects = meta.get("bbox_information", {}).get("objects", []) or []

    indices: list[int] = []
    if objects:
        for idx, obj in enumerate(objects, start=1):
            if isinstance(obj, dict) and obj.get("found") is False:
                continue
            indices.append(idx)
    else:
        indices = sorted(
            {
                int(p.stem.replace("bbox", ""))
                for p in sample_ref_dir.glob("bbox*.jpg")
                if p.stem[4:].isdigit()
            }
        )

    for idx in indices:
        img_path = sample_ref_dir / f"bbox{idx}.jpg"
        meta_path = sample_ref_dir / f"bbox{idx}.json"
        if not img_path.is_file() and meta_path.is_file():
            rm = load_json(meta_path)
            alt = Path(rm.get("output_image") or "")
            if alt.is_file():
                img_path = alt
        if not img_path.is_file():
            continue

        source_bbox = None
        label = None
        score = None
        area = 0.0
        if meta_path.is_file():
            rm = load_json(meta_path)
            source_bbox = rm.get("source_bbox")
            label = rm.get("object") or rm.get("detected_label")
            score = rm.get("detection_score")
            if source_bbox and len(source_bbox) == 4:
                x0, y0, x1, y1 = source_bbox
                area = max(0.0, (x1 - x0) * (y1 - y0))
        if area <= 0 and objects and 0 <= idx - 1 < len(objects) and isinstance(objects[idx - 1], dict):
            ob = objects[idx - 1]
            source_bbox = ob.get("bbox") or source_bbox
            label = ob.get("object") or ob.get("label") or label
            score = ob.get("score") or score
            if source_bbox and len(source_bbox) == 4:
                x0, y0, x1, y1 = source_bbox
                area = max(0.0, (x1 - x0) * (y1 - y0))

        refs.append(
            {
                "bbox_index": idx,
                "reference_image": str(img_path.resolve()),
                "reference_metadata": str(meta_path.resolve()) if meta_path.is_file() else None,
                "object": label,
                "detection_score": score,
                "source_bbox": source_bbox,
                "bbox_area": area,
            }
        )

    refs.sort(key=lambda r: (r["bbox_area"], r["bbox_index"]), reverse=True)
    return refs


def reference_bgr_and_mask(path: Path, white_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Load ref as BGR + FG mask (white/near-white → transparent)."""
    rgba = Image.open(path).convert("RGBA")
    arr = np.asarray(rgba)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    white = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)
    alpha = a.copy()
    alpha[white] = 0
    if alpha.max() == 0:
        alpha[:] = 255
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr, alpha


def fg_content_box(alpha: np.ndarray) -> tuple[float, float, float, float]:
    """Return (x0,y0,x1,y1,cx,cy,w,h)-like content box of non-zero alpha."""
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        h, w = alpha.shape[:2]
        return 0.0, 0.0, float(w), float(h)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    return x0, y0, x1, y1


def bbox_fit_scale_wh(
    content_w: float,
    content_h: float,
    bbox: list[float],
    fit: str,
    rot_deg: float = 0.0,
) -> float:
    """Uniform scale so a rotated content rect fits the source bbox."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    theta = np.deg2rad(float(rot_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    aabb_w = abs(c) * content_w + abs(s) * content_h
    aabb_h = abs(s) * content_w + abs(c) * content_h
    sx = bw / max(aabb_w, 1e-6)
    sy = bh / max(aabb_h, 1e-6)
    return float(min(sx, sy) if fit == "contain" else max(sx, sy))


def similarity_h_centered(
    scale: float,
    rot_deg: float,
    src_cx: float,
    src_cy: float,
    dst_cx: float,
    dst_cy: float,
) -> np.ndarray:
    """OpenCV similarity [a b tx; -b a ty] mapping src center → dst center."""
    theta = np.deg2rad(float(rot_deg))
    a = float(scale) * float(np.cos(theta))
    b = float(scale) * float(-np.sin(theta))
    tx = float(dst_cx) - (a * src_cx + b * src_cy)
    ty = float(dst_cy) - (-b * src_cx + a * src_cy)
    return np.array([[a, b, tx], [-b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, float] | None:
    """Return (scale, rot_deg) aligning src→dst via Umeyama (similarity, no reflect)."""
    if len(src) < 2:
        return None
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float((src_c ** 2).sum() / n)
    if var_s < 1e-12:
        return None
    cov = (dst_c.T @ src_c) / n  # 2x2
    u, d, vt = np.linalg.svd(cov)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        u[:, -1] *= -1.0
        d[-1] *= -1.0
    r = u @ vt
    scale = float(d.sum() / var_s)
    rot_deg = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    if not np.isfinite(scale) or scale < 1e-8:
        return None
    return scale, rot_deg


def _rotation_fixed_scale(
    src: np.ndarray,
    dst: np.ndarray,
    scale: float,
) -> float | None:
    """Kabsch rotation with predetermined uniform scale (src→dst)."""
    if len(src) < 2 or scale <= 0:
        return None
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = (src - mu_s) * float(scale)
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c
    u, _d, vt = np.linalg.svd(cov)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        u[:, -1] *= -1.0
    r = u @ vt
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def estimate_dift_rotation(
    kpts_ref: np.ndarray,
    kpts_gt: np.ndarray,
    *,
    expected_scale: float,
    min_inliers: int,
    ransac_thresh: float,
    scale_tol: float,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate rotation from DIFT matches with bbox scale locked.

    Quality gate: unconstrained Umeyama scale on the inliers must agree with
    ``expected_scale`` within ``scale_tol``. Rotation itself is always applied
    at the bbox scale (not the DIFT scale).
    """
    out: dict[str, Any] = {
        "accepted": False,
        "rotation_deg": 0.0,
        "dift_scale": None,
        "num_inliers": 0,
        "reject_reason": None,
    }
    if len(kpts_ref) < 3 or expected_scale <= 0:
        out["reject_reason"] = "insufficient_matches"
        return out

    rng = rng or np.random.default_rng(0)
    src = kpts_ref.astype(np.float64)
    dst = kpts_gt.astype(np.float64)
    n = len(src)
    best_inl = -1
    best_idx: np.ndarray | None = None
    best_rot = 0.0
    iters = min(4000, max(200, n * 20))
    s = float(expected_scale)

    for _ in range(iters):
        inds = rng.choice(n, size=min(3, n), replace=False)
        rot_deg = _rotation_fixed_scale(src[inds], dst[inds], s)
        if rot_deg is None:
            continue
        mu_s = src[inds].mean(axis=0)
        mu_d = dst[inds].mean(axis=0)
        theta = np.deg2rad(rot_deg)
        a = s * float(np.cos(theta))
        b = s * float(-np.sin(theta))
        tx = float(mu_d[0] - (a * mu_s[0] + b * mu_s[1]))
        ty = float(mu_d[1] - (-b * mu_s[0] + a * mu_s[1]))
        pred_x = a * src[:, 0] + b * src[:, 1] + tx
        pred_y = -b * src[:, 0] + a * src[:, 1] + ty
        err = np.hypot(pred_x - dst[:, 0], pred_y - dst[:, 1])
        inl_mask = err <= float(ransac_thresh)
        inl = int(inl_mask.sum())
        if inl > best_inl:
            best_inl = inl
            best_idx = np.where(inl_mask)[0]
            best_rot = float(rot_deg)

    out["num_inliers"] = int(max(best_inl, 0))
    if best_idx is None or best_inl < int(min_inliers):
        out["reject_reason"] = "few_inliers"
        return out

    free = _umeyama_similarity(src[best_idx], dst[best_idx])
    if free is None:
        out["reject_reason"] = "bad_scale"
        return out
    dift_scale, _ = free
    out["dift_scale"] = float(dift_scale)
    rel_err = abs(dift_scale / expected_scale - 1.0)
    out["scale_rel_err"] = float(rel_err)
    if rel_err > float(scale_tol):
        out["reject_reason"] = "scale_mismatch"
        return out

    rot_ref = _rotation_fixed_scale(src[best_idx], dst[best_idx], s)
    out["rotation_deg"] = float(rot_ref if rot_ref is not None else best_rot)
    out["accepted"] = True
    out["reject_reason"] = None
    return out


def warp_layer(
    ref_bgr: np.ndarray,
    ref_alpha: np.ndarray,
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
    warped_a = cv2.warpPerspective(
        ref_alpha,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, warped_a


def composite(canvas_bgr: np.ndarray, layer_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = canvas_bgr.astype(np.float32) * (1.0 - a) + layer_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_summary(H: np.ndarray, method: str) -> dict[str, float | str]:
    a, b = float(H[0, 0]), float(H[0, 1])
    scale = float(np.hypot(a, b))
    rot_deg = float(np.degrees(np.arctan2(-b, a)))
    return {
        "method": method,
        "scale_x": scale,
        "scale_y": scale,
        "rotation_deg": rot_deg,
        "tx": float(H[0, 2]),
        "ty": float(H[1, 2]),
    }


def process_sample(
    sample_id: str,
    *,
    theme_dir: Path,
    reference_dir: Path,
    bbox_meta_dir: Path,
    output_dir: Path,
    featurizer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    theme_path = theme_dir / f"{sample_id}.png"
    if not theme_path.is_file():
        raise FileNotFoundError(theme_path)

    gt = Image.open(theme_path).convert("RGB")
    out_w, out_h = gt.size
    bg = (255, 255, 255) if args.canvas_bg == "white" else (0, 0, 0)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = bg

    refs = collect_refs(sample_id, reference_dir, bbox_meta_dir)
    placements_out: list[dict[str, Any]] = []

    for i, ref_info in enumerate(refs):
        ref_path = Path(ref_info["reference_image"])
        ref_pil = Image.open(ref_path).convert("RGBA")
        rw, rh = ref_pil.size
        bbox = ref_info.get("source_bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"{sample_id} bbox{ref_info['bbox_index']}: missing source_bbox")

        x0, y0, x1, y1 = [float(v) for v in bbox]
        bbox_cx = 0.5 * (x0 + x1)
        bbox_cy = 0.5 * (y0 + y1)

        ref_bgr, ref_alpha = reference_bgr_and_mask(ref_path, args.white_threshold)
        fx0, fy0, fx1, fy1 = fg_content_box(ref_alpha)
        content_w = max(1.0, fx1 - fx0)
        content_h = max(1.0, fy1 - fy0)
        ref_cx = 0.5 * (fx0 + fx1)
        ref_cy = 0.5 * (fy0 + fy1)
        # Baseline (no-rotation) scale from FG content — validates DIFT scale.
        s_bbox0 = bbox_fit_scale_wh(content_w, content_h, bbox, args.fit, rot_deg=0.0)

        corr = match_ref_to_gt_correspondences(
            featurizer,
            gt,
            ref_pil,
            prompt=str(ref_info.get("object") or ""),
            img_size=int(args.img_size),
            t=DEFAULT_T,
            up_ft_index=DEFAULT_UP_FT_INDEX,
            ensemble_size=int(args.ensemble),
            num_queries=int(args.num_queries),
            seed=i + 17,
            gt_bbox=[x0, y0, x1, y1],
        )
        kpts_ref = corr["kpts_ref"]
        kpts_gt = corr["kpts_gt"]

        rot_info: dict[str, Any] = {
            "accepted": False,
            "rotation_deg": 0.0,
            "num_inliers": 0,
            "reject_reason": "skipped",
        }
        method = "bbox_place"
        rot_deg = 0.0

        if len(kpts_ref) >= int(args.min_matches):
            rot_info = estimate_dift_rotation(
                kpts_ref,
                kpts_gt,
                expected_scale=s_bbox0,
                min_inliers=int(args.min_inliers),
                ransac_thresh=float(args.ransac_thresh),
                scale_tol=float(args.scale_tol),
                rng=np.random.default_rng(i + 91),
            )
            if rot_info["accepted"]:
                rot_deg = float(rot_info["rotation_deg"])
                method = "bbox_place+dift_rot"

        # Always bbox-anchored scale (shrink if rotation inflates AABB); DIFT
        # may supply rotation only.
        s_bbox = bbox_fit_scale_wh(content_w, content_h, bbox, args.fit, rot_deg=rot_deg)
        H = similarity_h_centered(s_bbox, rot_deg, ref_cx, ref_cy, bbox_cx, bbox_cy)

        layer, alpha = warp_layer(ref_bgr, ref_alpha, H, out_w, out_h)
        canvas = composite(canvas, layer, alpha)

        summary = transform_summary(H, method)
        placements_out.append(
            {
                **{
                    k: ref_info[k]
                    for k in (
                        "bbox_index",
                        "object",
                        "detection_score",
                        "source_bbox",
                        "reference_image",
                    )
                },
                "num_matches": int(len(kpts_ref)),
                "num_inliers": int(rot_info.get("num_inliers") or 0),
                "dift_confidence": float(corr.get("confidence") or 0.0),
                "mean_sim": float(corr.get("mean_sim") or 0.0),
                "kept_mean_sim": float(corr.get("kept_mean_sim") or 0.0),
                "bbox_scale": float(s_bbox),
                "bbox_scale_unrotated": float(s_bbox0),
                "dift_scale": rot_info.get("dift_scale"),
                "scale_rel_err": rot_info.get("scale_rel_err"),
                "dift_rot_accepted": bool(rot_info.get("accepted")),
                "dift_reject_reason": rot_info.get("reject_reason"),
                "search_bbox": corr.get("search_bbox"),
                "homography": H.tolist(),
                **summary,
            }
        )

    out_img_dir = output_dir / "images"
    out_meta_dir = output_dir / "metadata"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_img_dir / f"{sample_id}.png"
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

    meta = {
        "id": sample_id,
        "theme_image": str(theme_path.resolve()),
        "canvas_image": str(out_path.resolve()),
        "image_width": out_w,
        "image_height": out_h,
        "canvas_bg": args.canvas_bg,
        "method": "bbox_anchored_dift_rotation",
        "fit": args.fit,
        "scale_tol": float(args.scale_tol),
        "num_references": len(placements_out),
        "placements": placements_out,
        "dift": {
            "img_size": int(args.img_size),
            "ensemble": int(args.ensemble),
            "num_queries": int(args.num_queries),
            "t": DEFAULT_T,
            "up_ft_index": DEFAULT_UP_FT_INDEX,
            "min_inliers": int(args.min_inliers),
            "ransac_thresh": float(args.ransac_thresh),
        },
    }
    write_json(out_meta_dir / f"{sample_id}.json", meta)

    if args.save_debug:
        dbg_dir = output_dir / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        theme_bgr = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        panel = np.concatenate([theme_bgr, canvas], axis=1)
        cv2.imwrite(str(dbg_dir / f"{sample_id}_theme_canvas.jpg"), panel)

    return meta


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    theme_dir = args.theme_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    bbox_meta_dir = args.bbox_metadata_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = list_sample_ids(theme_dir, reference_dir, args.theme)
    if args.limit and args.limit > 0:
        sample_ids = sample_ids[: int(args.limit)]
    if not sample_ids:
        raise SystemExit(f"No samples found under {theme_dir} with refs in {reference_dir}")

    print(f"Samples: {len(sample_ids)} | GPU={args.gpu_id} | out={output_dir}")
    print(f"SD model: {args.sd_model}", flush=True)
    os.environ["DIFT_SD_MODEL"] = str(args.sd_model)
    featurizer = get_featurizer(str(args.sd_model))
    print("DIFT featurizer ready", flush=True)

    ok = skipped = failed = 0
    t0 = time.perf_counter()
    for sid in tqdm(sample_ids, desc="canvas_dift"):
        out_img = output_dir / "images" / f"{sid}.png"
        if args.skip_existing and out_img.is_file():
            skipped += 1
            continue
        try:
            process_sample(
                sid,
                theme_dir=theme_dir,
                reference_dir=reference_dir,
                bbox_meta_dir=bbox_meta_dir,
                output_dir=output_dir,
                featurizer=featurizer,
                args=args,
            )
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {sid}: {exc}", file=sys.stderr)

    print(
        f"Done ok={ok} skipped={skipped} failed={failed} "
        f"elapsed={time.perf_counter() - t0:.1f}s → {output_dir}"
    )


if __name__ == "__main__":
    main()
