import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

# --- Config ---
model_name = "yolo26n.pt"                                  # or yolov26s.pt, yolov26m.pt, etc.
image_dir  = "output/images/"                       # folder with ~5000 images
output_csv = "output/bbox_results/yolo26_detections.csv" # single CSV output
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
image_paths = list(image_dir.rglob("*.jpg")) + \
              list(image_dir.rglob("*.jpeg")) + \
              list(image_dir.rglob("*.png")) + \
              list(image_dir.rglob("*.webp"))

print(f"Found {len(image_paths)} images, running YOLO26 on {device!r}...")

data = []

for im_path in tqdm(image_paths):
    im_path = str(im_path)

    results = model(im_path, conf=conf_thr, device=device, verbose=False)

    for result in results:
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                data.append({
                    "image_path": os.path.basename(im_path),
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
print(f"Unique images: {df['image_path'].nunique()}")
