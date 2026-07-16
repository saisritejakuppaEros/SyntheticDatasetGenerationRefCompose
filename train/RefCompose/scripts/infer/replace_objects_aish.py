import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
import numpy as np
import os

# Load Grounding DINO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

def get_boxes(image, text_labels):
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )[0]
    return results

canvas_path = "/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/flux_server/output_bg_removed/flux_client_run_aish/bg_removed_cultural_scene_aish_var02_seed43.png"
lantern_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/lantern.png"
trident_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/tridentt.png"

# Load background-removed objects
objects_img = Image.open(canvas_path).convert("RGBA")
canvas_w, canvas_h = objects_img.size

# Create a NEW canvas with a stone-grey background (RGB: 70, 65, 60)
canvas = Image.new("RGBA", (canvas_w, canvas_h), (70, 65, 60, 255))
canvas.paste(objects_img, (0, 0), mask=objects_img)

# Find lantern and trident
results = get_boxes(objects_img.convert("RGB"), ["a lantern. a trident. a spear."])

boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]

def rgba_cover_to_box(src: Image.Image, box_w: int, box_h: int) -> Image.Image:
    src = src.convert("RGBA")
    iw, ih = src.size
    scale = max(box_w / iw, box_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - box_w) // 2
    top = (nh - box_h) // 2
    right = left + box_w
    bottom = top + box_h
    return resized.crop((left, top, right, bottom))

lantern_img = Image.open(lantern_path).convert("RGBA")
trident_img = Image.open(trident_path).convert("RGBA")

draw = ImageDraw.Draw(canvas)

for box, label, score in zip(boxes, labels, scores):
    x0, y0, x1, y1 = map(int, box.tolist())
    w, h = x1 - x0, y1 - y0
    
    if "lantern" in label:
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        fitted = rgba_cover_to_box(lantern_img, w, h)
        canvas.paste(fitted, (x0, y0), mask=fitted)
    elif "trident" in label or "spear" in label:
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        fitted = rgba_cover_to_box(trident_img, w, h)
        canvas.paste(fitted, (x0, y0), mask=fitted)

canvas.save("replaced_canvas_aish_stone.png")
print("Saved replaced_canvas_aish_stone.png with stone background")
