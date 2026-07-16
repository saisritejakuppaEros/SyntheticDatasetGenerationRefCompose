#!/usr/bin/env python3
"""
CSV-driven multiview generation: filter YOLO boxes by min side length, crop from full
images, run Qwen + multi-angle LoRA. Outputs: out_dir / image_stem / crop_no / viewno.png

Parallel mode: loads one pipeline per worker (default: 3 workers per GPU × 2 GPUs).
Uses multiprocessing (spawn) so each process owns its CUDA context. Main process shows
tqdm with ETA over completed crops.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from tqdm import tqdm

# --- defaults (override via CLI) ---
DEFAULT_QWEN_MODEL_PATH = "/mnt/data0/priyadharsan/Week16/Rnd/model/qwen-image-edit-2511"
DEFAULT_LORA_PATH = "/mnt/data0/priyadharsan/Week16/Rnd/Canvas/models"
DEFAULT_LORA_FILE = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
DEFAULT_GPUS = "3,5"
DEFAULT_WORKERS_PER_GPU = 3

AZIMUTHS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]
ELEVATIONS = ["low-angle shot", "eye-level shot", "elevated shot", "high-angle shot"]
DISTANCES = ["close-up", "medium shot", "wide shot"]


def get_random_multiview_prompt() -> str:
    while True:
        azi = random.choice(AZIMUTHS)
        ele = random.choice(ELEVATIONS)
        dist = random.choice(DISTANCES)
        if azi == "front view" and ele == "eye-level shot" and dist == "medium shot":
            continue
        return f"<sks> {azi} {ele} {dist}"


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


def clamp_xyxy(
    x1: float, y1: float, x2: float, y2: float, iw: int, ih: int
) -> tuple[int, int, int, int]:
    x1i = int(round(x1))
    y1i = int(round(y1))
    x2i = int(round(x2))
    y2i = int(round(y2))
    x1i = max(0, min(x1i, iw - 1))
    y1i = max(0, min(y1i, ih - 1))
    x2i = max(x1i + 1, min(x2i, iw))
    y2i = max(y1i + 1, min(y2i, ih))
    return x1i, y1i, x2i, y2i


def load_qwen_pipeline(
    qwen_model: str,
    lora_path: str,
    lora_file: str,
    lora_weight: float,
    device: torch.device,
) -> QwenImageEditPlusPipeline:
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_model, torch_dtype=torch.bfloat16
    ).to(device)
    pipeline.load_lora_weights(lora_path, weight_name=lora_file, adapter_name="multiview")
    pipeline.set_adapters(["multiview"], adapter_weights=[lora_weight])
    return pipeline


def _process_crop_job(
    pipeline: QwenImageEditPlusPipeline,
    job: dict[str, Any],
) -> None:
    """Run all pending views for one crop (in-process)."""
    src = Path(job["src"])
    full = Image.open(src).convert("RGB")
    iw, ih = full.size
    x1, y1, x2, y2 = clamp_xyxy(
        job["x1"], job["y1"], job["x2"], job["y2"], iw, ih
    )
    if (x2 - x1) <= job["min_side"] or (y2 - y1) <= job["min_side"]:
        return
    crop_pil = full.crop((x1, y1, x2, y2))
    out_crop_dir = Path(job["out_crop_dir"])
    out_crop_dir.mkdir(parents=True, exist_ok=True)
    class_name = job["class_name"]
    count = int(job["count"])
    num_inference_steps = int(job["num_inference_steps"])
    true_cfg_scale = float(job["true_cfg_scale"])

    for v in job["pending_views"]:
        random_angle = get_random_multiview_prompt()
        prompt = (
            f"{random_angle} of the individual 3D {class_name}, isolated on black background. "
            "High quality textures."
        )
        with torch.inference_mode():
            output = pipeline(
                image=crop_pil,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=torch.manual_seed(random.randint(0, 1000000)),
            ).images[0]
        out_path = out_crop_dir / f"{v}.png"
        output.save(out_path)


def _worker_loop(
    gpu_id: int,
    worker_slot: int,
    args_dict: dict[str, Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Child process: load one pipeline on gpu_id, drain task_queue until None."""
    device = torch.device(f"cuda:{gpu_id}")
    if not torch.cuda.is_available():
        result_queue.put(
            {"ok": False, "error": "CUDA not available in worker", "gpu_id": gpu_id}
        )
        return
    if gpu_id >= torch.cuda.device_count():
        result_queue.put(
            {
                "ok": False,
                "error": f"cuda:{gpu_id} not found (device_count={torch.cuda.device_count()})",
                "gpu_id": gpu_id,
            }
        )
        return

    t0 = time.perf_counter()
    try:
        pipeline = load_qwen_pipeline(
            args_dict["qwen_model"],
            args_dict["lora_path"],
            args_dict["lora_file"],
            args_dict["lora_weight"],
            device,
        )
    except Exception as e:
        result_queue.put({"ok": False, "error": repr(e), "gpu_id": gpu_id})
        return

    load_s = time.perf_counter() - t0
    result_queue.put(
        {
            "ok": True,
            "ready": True,
            "gpu_id": gpu_id,
            "worker_slot": worker_slot,
            "load_s": load_s,
        }
    )

    while True:
        job = task_queue.get()
        if job is None:
            break
        label = job.get("label", "")
        try:
            _process_crop_job(pipeline, job)
            result_queue.put({"ok": True, "done": True, "label": label})
        except Exception as e:
            result_queue.put({"ok": False, "done": True, "label": label, "error": repr(e)})


def _parse_gpus(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--gpus must list at least one device id, e.g. 3,5")
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiview crops from YOLO CSV + full images.")
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
        required=True,
        help="Output root: out_dir / image_stem / crop_no / viewno.png",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of multiview images per crop.")
    parser.add_argument(
        "--min-side",
        type=float,
        default=200.0,
        help="Keep boxes only if both width and height are strictly greater than this.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most this many crops (after filter/sort), for dry runs.",
    )
    parser.add_argument("--qwen-model", type=Path, default=Path(DEFAULT_QWEN_MODEL_PATH))
    parser.add_argument("--lora-path", type=Path, default=Path(DEFAULT_LORA_PATH))
    parser.add_argument("--lora-file", type=str, default=DEFAULT_LORA_FILE)
    parser.add_argument("--lora-weight", type=float, default=0.9)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.5)
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help=f"Comma-separated CUDA device indices (default: {DEFAULT_GPUS}).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
        help=f"Independent pipeline replicas per GPU (default: {DEFAULT_WORKERS_PER_GPU}).",
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

    df = load_and_filter_detections(csv_path, args.min_side)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    if len(df) == 0:
        print("No rows after filtering; nothing to do.")
        return

    jobs: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        image_path = row.image_path
        stem = Path(image_path).stem
        crop_no = int(row.crop_no)
        class_name = str(row.class_name)

        src = images_dir / image_path
        if not src.is_file():
            print(f"skip missing image: {src}", file=sys.stderr)
            continue

        out_crop_dir = args.out_dir / stem / str(crop_no)
        pending_views = [
            v for v in range(1, args.count + 1) if not (out_crop_dir / f"{v}.png").is_file()
        ]
        if not pending_views:
            continue

        jobs.append(
            {
                "label": f"{stem}/{crop_no}",
                "src": str(src),
                "x1": float(row.x1),
                "y1": float(row.y1),
                "x2": float(row.x2),
                "y2": float(row.y2),
                "class_name": class_name,
                "out_crop_dir": str(out_crop_dir.resolve()),
                "pending_views": pending_views,
                "count": args.count,
                "min_side": args.min_side,
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
            }
        )

    if not jobs:
        print("Nothing to do (all views already exist or no valid rows).")
        return

    num_workers = len(gpu_ids) * wpg
    args_dict = {
        "qwen_model": str(args.qwen_model.resolve()),
        "lora_path": str(args.lora_path.resolve()),
        "lora_file": args.lora_file,
        "lora_weight": args.lora_weight,
    }

    print(
        f"Planned: {len(jobs)} crop job(s), {num_workers} worker process(es) "
        f"({wpg} pipeline(s) per GPU on {gpu_ids}).",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    workers: list[mp.Process] = []
    wi = 0
    for gpu_id in gpu_ids:
        for _ in range(wpg):
            p = ctx.Process(
                target=_worker_loop,
                args=(gpu_id, wi, args_dict, task_queue, result_queue),
                name=f"mv-worker-g{gpu_id}-s{wi}",
            )
            p.start()
            workers.append(p)
            wi += 1

    # Wait for each worker to load the model (or fail fast).
    ready_ok = 0
    load_times: list[float] = []
    for _ in range(num_workers):
        msg = result_queue.get()
        if not msg.get("ok"):
            err = msg.get("error", "unknown")
            for proc in workers:
                proc.terminate()
            raise SystemExit(f"Worker failed during pipeline load: {err}")
        if msg.get("ready"):
            ready_ok += 1
            load_times.append(float(msg.get("load_s", 0.0)))
            print(
                f"  Worker on cuda:{msg['gpu_id']} slot {msg['worker_slot']} ready "
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
        desc="multiview crops",
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

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    elapsed = time.perf_counter() - t_start
    print(
        f"Finished {len(jobs)} crop(s) in {elapsed:.1f}s "
        f"({elapsed / max(len(jobs), 1):.2f}s per crop avg).",
        flush=True,
    )
    if load_times:
        print(f"Per-worker model load times (s): min={min(load_times):.1f} max={max(load_times):.1f}")
    if errors:
        print(f"{len(errors)} job(s) reported errors:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()


# python stage3_multiview.py \
#   --csv output/bbox_results/yolo26_detections.csv \
#   --image-path output/images \
#   --out-dir output/multiview_out \
#   --gpus 3,5 --workers-per-gpu 3
