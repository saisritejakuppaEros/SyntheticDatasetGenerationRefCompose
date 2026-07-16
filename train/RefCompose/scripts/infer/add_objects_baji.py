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

canvas_path = "/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/flux_server/output_bg_removed/flux_client_run_baji/bg_removed_cultural_scene_baji_var02_seed43.png"
lota_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/lota.png"
diya_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/diya.png"

# Load background-removed objects
objects_img = Image.open(canvas_path).convert("RGBA")
canvas_w, canvas_h = objects_img.size

# Create a NEW canvas with a stone-grey background (RGB: 70, 65, 60)
canvas = Image.new("RGBA", (canvas_w, canvas_h), (70, 65, 60, 255))
canvas.paste(objects_img, (0, 0), mask=objects_img)

# Check for existing pot/lamp to replace
results = get_boxes(objects_img.convert("RGB"), ["a pot. a lamp."])
boxes = results["boxes"]
labels = results["labels"]

lota_img = Image.open(lota_path).convert("RGBA")
diya_img = Image.open(diya_path).convert("RGBA")

draw = ImageDraw.Draw(canvas)

if len(boxes) > 0:
    print("Found existing objects to replace:", labels)
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = map(int, box.tolist())
        w, h = x1 - x0, y1 - y0
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        if "pot" in label:
            lw, lh = lota_img.size
            new_lw = w
            new_lh = int(lh * (new_lw / lw))
            fitted = lota_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
            canvas.paste(fitted, (x0, y1 - new_lh), mask=fitted)
        elif "lamp" in label:
            lw, lh = diya_img.size
            new_lw = w
            new_lh = int(lh * (new_lw / lw))
            fitted = diya_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
            canvas.paste(fitted, (x0, y1 - new_lh), mask=fitted)
else:
    print("No existing objects found. Adding them to the floor.")
    # Add lota on the bottom right
    lw, lh = lota_img.size
    new_lh = 200
    new_lw = int(lw * (new_lh / lh))
    fitted_lota = lota_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
    canvas.paste(fitted_lota, (canvas_w - new_lw - 100, canvas_h - new_lh - 50), mask=fitted_lota)
    
    # Add diya on the bottom left
    dw, dh = diya_img.size
    new_dh = 120
    new_dw = int(dw * (new_dh / dh))
    fitted_diya = diya_img.resize((new_dw, new_dh), Image.Resampling.LANCZOS)
    canvas.paste(fitted_diya, (100, canvas_h - new_dh - 50), mask=fitted_diya)

canvas.save("replaced_canvas_baji_stone.png")
print("Saved replaced_canvas_baji_stone.png")
