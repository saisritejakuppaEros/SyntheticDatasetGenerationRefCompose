import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

# --- Config ---
model_name = "yolo26n.pt"                                  # or yolov26s.pt, yolov26m.pt, etc.
image_dir  = "/mnt/data0/teja/multiref_image/model_training_dataset/hd_1280x720/"                       # folder with ~5000 images
output_csv = "output/bbox_results/yolo26_detections.csv" # single CSV output
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
conf_thr   = 0.25                                           # min confidence threshold
# Prefer CUDA when it actually works; old drivers / no GPU -> CPU (avoids ValueError on device=0).
_device_pref = os.environ.get("YOLO_DEVICE", "cuda").lower()
if _device_pref in ("cuda", "gpu") and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# --- Setup ---
model = YOLO(model_name)
image_dir = Path(image_dir)
image_paths = list(image_dir.rglob("*/*.jpg")) + \
              list(image_dir.rglob("*/*.jpeg")) + \
              list(image_dir.rglob("*/*.png")) + \
              list(image_dir.rglob("*/*.webp"))

print(f"Found {len(image_paths)} images, running YOLO26 on {device!r}...")

data = []
skipped = 0

for im_path in tqdm(image_paths):
    im_path = Path(im_path)
    im_str = str(im_path)

    if not im_path.is_file():
        skipped += 1
        tqdm.write(f"[skip] missing: {im_str}", file=sys.stderr)
        continue
    try:
        if im_path.stat().st_size == 0:
            skipped += 1
            tqdm.write(f"[skip] empty file: {im_str}", file=sys.stderr)
            continue
    except OSError as e:
        skipped += 1
        tqdm.write(f"[skip] stat {im_str}: {e}", file=sys.stderr)
        continue

    try:
        results = model(im_str, conf=conf_thr, device=device, verbose=False)
    except cv2.error as e:
        skipped += 1
        tqdm.write(f"[skip] OpenCV {im_str}: {e}", file=sys.stderr)
        continue
    except Exception as e:  # noqa: BLE001
        skipped += 1
        tqdm.write(f"[skip] {im_str}: {e}", file=sys.stderr)
        continue

    for result in results:
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                data.append({
                    "image_path": im_path.name,
                    "image_abspath": str(im_path.resolve()),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                })

# --- Save CSV ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"\nDone. {len(df)} detections saved to {output_csv}")
if skipped:
    print(f"Skipped {skipped} unreadable / failed image(s).")
if len(df):
    print(f"Unique images (basename): {df['image_path'].nunique()}")
    print(f"Unique images (abspath): {df['image_abspath'].nunique()}")
