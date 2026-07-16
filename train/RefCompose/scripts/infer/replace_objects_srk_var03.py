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

canvas_path = "/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/flux_server/output_bg_removed/flux_client_run/bg_removed_cultural_scene_var03_seed44 (1).png"
lota_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/lota1.png"
diya_path = "/mnt/data0/teja/research_multiref/lora_training_v2_parth/ads3/bg_removed/diya.png"
actor_path = "/mnt/data0/parth/images/ads_nobg/bahubali/prabhas.png"

# Load background-removed objects
objects_img = Image.open(canvas_path).convert("RGBA")
canvas_w, canvas_h = objects_img.size

# Create a NEW canvas with a stone-grey background (RGB: 70, 65, 60)
canvas = Image.new("RGBA", (canvas_w, canvas_h), (70, 65, 60, 255))
canvas.paste(objects_img, (0, 0), mask=objects_img)

# Find pot, lamp, and standing actor
results = get_boxes(objects_img.convert("RGB"), ["a copper pot. a brass lamp. a standing man."])

boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]

print("Found boxes:", list(zip(labels, scores.tolist(), boxes.tolist())))

lota_img = Image.open(lota_path).convert("RGBA")
diya_img = Image.open(diya_path).convert("RGBA")
actor_img = Image.open(actor_path).convert("RGBA")

draw = ImageDraw.Draw(canvas)

for box, label, score in zip(boxes, labels, scores):
    x0, y0, x1, y1 = map(int, box.tolist())
    w, h = x1 - x0, y1 - y0
    
    if "pot" in label:
        # Clear pot area (stone-grey background)
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        lw, lh = lota_img.size
        new_lw = w
        new_lh = int(lh * (new_lw / lw))
        fitted = lota_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
        # Paste at the bottom of the old pot area
        canvas.paste(fitted, (x0, y1 - new_lh), mask=fitted)
        
    elif "lamp" in label:
        # Clear lamp area
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        lw, lh = diya_img.size
        new_lw = w
        new_lh = int(lh * (new_lw / lw))
        fitted = diya_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x0, y1 - new_lh), mask=fitted)

    elif "standing" in label:
        # Clear actor area
        draw.rectangle([x0, y0, x1, y1], fill=(70, 65, 60, 255))
        aw, ah = actor_img.size
        new_lh = h
        new_lw = int(aw * (new_lh / ah))
        fitted = actor_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x0 + (w - new_lw) // 2, y0), mask=fitted)

# Save locally
canvas.save("replaced_canvas_srk_var03_stone.png")
print("Saved replaced_canvas_srk_var03_stone.png")

# Save in the user-specified output folder
output_dir = "/mnt/data0/teja/research_multiref/lora_training_v2/infer/parth_output/cultural/mix"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "replaced_canvas_srk_var03_stone.png")
canvas.save(output_path)
print(f"Saved result to {output_path}")
