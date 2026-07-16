#!/usr/bin/env python3
"""
CSV-driven Hunyuan3D multiview: filter YOLO boxes by min side length (default: width and
height both strictly > 100), crop from full images, run Hunyuan3D-2.1/infer.py logic per
crop. By default processes every qualifying box in the CSV across all images (--max-samples 0).

Parallel mode (default): spawns (workers-per-gpu × len(gpus)) processes; each process sets
CUDA_VISIBLE_DEVICES to one physical GPU and loads its own shape + texture + RMBG stacks.
Uses multiprocessing (spawn) + shared task queue for load-balanced crops. Main process
shows tqdm with ETA.

By default, crops are passed through RMBG-1.4 before mesh gen; use --no-remove-bg to skip.

Outputs per crop: out_dir / image_stem / crop_no / (infer artifacts, including
multiviews_textured/view_XX.png). Saves the crop as crop_input.png for traceability.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HUNYUAN_ROOT = SCRIPT_DIR / "multi_view_images" / "Hunyuan3D-2.1"
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "output" / "Hunyuan3d_multiview"
DEFAULT_GPUS = "3,5"
DEFAULT_WORKERS_PER_GPU = 3


def load_and_filter_detections(csv_path: Path, min_side: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("x1", "y1", "x2", "y2", "image_path", "class_name", "confidence"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column: {col}")
    df = df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df = df[(df["w"] > min_side) & (df["h"] > min_side)]
    df = df.sort_values(
        by=["image_path", "confidence", "x1", "y1", "x2", "y2"],
        ascending=[True, False, True, True, True, True],
    )
    df["crop_no"] = df.groupby("image_path", sort=False).cumcount()
    return df


def _parse_gpus(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--gpus must list at least one device id, e.g. 3,5")
    return [int(p) for p in parts]


def main() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from hunyuan_stage3_worker import worker_main

    parser = argparse.ArgumentParser(
        description="Multiview crops from YOLO CSV + Hunyuan3D-2.1 (multi-GPU worker pool)."
    )
    parser.add_argument(
        "--csv",
        "--csv-path",
        type=Path,
        required=True,
        dest="csv_path",
        help="Path to YOLO detections CSV (image_path, x1..y2, class_name, confidence, ...).",
    )
    parser.add_argument(
        "--image-path",
        "--images-dir",
        type=Path,
        required=True,
        dest="image_path",
        help="Directory containing full-resolution images (basenames match CSV image_path column).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output root (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--min-side",
        type=float,
        default=100.0,
        help="Keep boxes only if both width and height are strictly greater than this.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Process at most this many crops after filter/sort (default: 0 = all).",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=4,
        help="Number of textured multi-view renders.",
    )
    parser.add_argument(
        "--elev-min",
        type=float,
        default=-25.0,
        help="Minimum camera elevation (deg) for multiview grid.",
    )
    parser.add_argument(
        "--elev-max",
        type=float,
        default=25.0,
        help="Maximum camera elevation (deg) for multiview grid.",
    )
    parser.add_argument(
        "--elev-steps",
        type=int,
        default=3,
        help="Elevation levels from elev-min to elev-max (used only with --grid-multiview).",
    )
    parser.set_defaults(orbit_fixed_elev=True)
    parser.add_argument(
        "--grid-multiview",
        action="store_false",
        dest="orbit_fixed_elev",
        help="Use azimuth×elevation lattice instead of a fixed-elevation azimuth orbit (infer.py default).",
    )
    parser.add_argument(
        "--fixed-elev",
        type=float,
        default=0.0,
        help="Camera elevation in degrees for the orbit ring (default: 0). Ignored with --grid-multiview.",
    )
    parser.add_argument(
        "--mesh-up",
        type=str,
        choices=("y_to_z", "longest_z", "none"),
        default="y_to_z",
        help="Stand mesh along +Z before rendering (see infer.py).",
    )
    parser.add_argument(
        "--paint-views",
        type=int,
        default=6,
        help="max_num_view for texture pipeline (6–9 typical).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Internal paint resolution (512 or 768).",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=768,
        help="Output PNG size for textured views.",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Skip texturing; gray matplotlib previews only (no paint pipeline load).",
    )
    parser.add_argument(
        "--hunyuan-root",
        type=Path,
        default=DEFAULT_HUNYUAN_ROOT,
        help="Directory containing infer.py and hy3dshape/hy3dpaint.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a crop if multiviews_textured/view_00.png already exists under its out dir.",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_false",
        dest="remove_bg",
        help="Skip RMBG-1.4 background removal on each crop before Hunyuan (default: remove background).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help=f"Comma-separated physical CUDA indices (default: {DEFAULT_GPUS}). Each worker sets CUDA_VISIBLE_DEVICES to one id.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
        help=f"Independent full stacks (shape+texture+RMBG) per GPU (default: {DEFAULT_WORKERS_PER_GPU}).",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    images_dir = args.image_path.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"image-path must be a directory of images, not found: {images_dir}")

    gpu_ids = _parse_gpus(args.gpus)
    wpg = args.workers_per_gpu
    if wpg < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")

    hunyuan_root = args.hunyuan_root.resolve()

    df = load_and_filter_detections(csv_path, args.min_side)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    if len(df) == 0:
        print("No rows after filtering; nothing to do.")
        return

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    infer_ns: dict[str, Any] = {
        "views": args.views,
        "elev_min": args.elev_min,
        "elev_max": args.elev_max,
        "elev_steps": args.elev_steps,
        "mesh_up": args.mesh_up,
        "render_size": args.render_size,
        "geometry_only": args.geometry_only,
        "orbit_fixed_elev": args.orbit_fixed_elev,
        "fixed_elev": args.fixed_elev,
    }

    args_dict: dict[str, Any] = {
        "hunyuan_root": str(hunyuan_root),
        "geometry_only": args.geometry_only,
        "paint_views": args.paint_views,
        "resolution": args.resolution,
        "remove_bg": args.remove_bg,
        "infer_ns": infer_ns,
    }

    jobs: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        image_path = row.image_path
        stem = Path(image_path).stem
        crop_no = int(row.crop_no)

        src = images_dir / image_path
        if not src.is_file():
            print(f"skip missing image: {src}", file=sys.stderr)
            continue

        out_crop_dir = out_root / stem / str(crop_no)
        views_dir = out_crop_dir / "multiviews_textured"
        if args.skip_existing and (views_dir / "view_00.png").is_file():
            print(f"skip existing: {views_dir}", file=sys.stderr)
            continue

        jobs.append(
            {
                "label": f"{stem}/{crop_no}",
                "src": str(src),
                "x1": float(row.x1),
                "y1": float(row.y1),
                "x2": float(row.x2),
                "y2": float(row.y2),
                "out_crop_dir": str(out_crop_dir.resolve()),
                "min_side": args.min_side,
            }
        )

    if not jobs:
        print("Nothing to do (no jobs after skip-existing / missing files).")
        return

    num_workers = len(gpu_ids) * wpg
    print(
        f"Queued {len(jobs)} crop job(s); {num_workers} worker process(es) "
        f"({wpg} stack(s) per GPU on physical GPUs {gpu_ids}).",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    workers: list[mp.Process] = []
    wi = 0
    for gpu_id in gpu_ids:
        for _ in range(wpg):
            p = ctx.Process(
                target=worker_main,
                args=(gpu_id, wi, args_dict, task_queue, result_queue),
                name=f"hunyuan-w-g{gpu_id}-s{wi}",
            )
            p.start()
            workers.append(p)
            wi += 1

    ready_ok = 0
    load_times: list[float] = []
    for _ in range(num_workers):
        msg = result_queue.get()
        if not msg.get("ok"):
            err = msg.get("error", "unknown")
            for proc in workers:
                proc.terminate()
            raise SystemExit(f"Worker failed during model load: {err}")
        if msg.get("ready"):
            ready_ok += 1
            load_times.append(float(msg.get("load_s", 0.0)))
            print(
                f"  Worker physical GPU {msg['gpu_id']} slot {msg['worker_slot']} ready "
                f"(load {msg.get('load_s', 0):.1f}s)",
                flush=True,
            )

    if ready_ok != num_workers:
        for proc in workers:
            proc.terminate()
        raise SystemExit("Unexpected worker ready count.")

    for job in jobs:
        task_queue.put(job)
    for _ in range(num_workers):
        task_queue.put(None)

    t_start = time.perf_counter()
    errors: list[str] = []
    with tqdm(
        total=len(jobs),
        desc="hunyuan_multiview",
        unit="crop",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for _ in range(len(jobs)):
            msg = result_queue.get()
            if msg.get("done"):
                pbar.update(1)
                if not msg.get("ok"):
                    errors.append(f"{msg.get('label', '?')}: {msg.get('error', '')}")
            else:
                pbar.write(f"unexpected message: {msg}")

    for proc in workers:
        proc.join(timeout=600)
        if proc.is_alive():
            proc.terminate()

    elapsed = time.perf_counter() - t_start
    print(
        f"Finished {len(jobs)} crop(s) in {elapsed:.1f}s "
        f"({elapsed / max(len(jobs), 1):.2f}s per crop avg).",
        flush=True,
    )
    if load_times:
        print(
            f"Per-worker model load times (s): min={min(load_times):.1f} max={max(load_times):.1f}"
        )
    if errors:
        print(f"{len(errors)} job(s) reported errors:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()

# Example:
#   python dataset_preparation/stage3_hunyuan_multiview.py \
#     --csv dataset_preparation/output/bbox_results/yolo26_detections.csv \
#     --image-path dataset_preparation/output/images \
#     --gpus 3,5 --workers-per-gpu 3


# python stage3_hunyuan_multiview.py   --csv output/bbox_results/yolo26_detections.csv   --image-path output/images   --gpus 3,5   --workers-per-gpu 3