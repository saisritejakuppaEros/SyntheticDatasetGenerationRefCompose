import pandas as pd
import glob
import os
import io
from PIL import Image

data_dir = "/mnt/data0/priyadharsan/Week16/Rnd/Dataset/DenseLayout/data"
parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

def find_and_extract_dl1():
    for f in parquet_files:
        print(f"Checking {f}...")
        df = pd.read_parquet(f)
        row = df[df['id'] == 'dl_1']
        if not row.empty:
            print(f"Found dl_1 in {f}!")
            return row.iloc[0]
    return None

dl1_data = find_and_extract_dl1()
if dl1_data is not None:
    # Extract image
    img_dict = dl1_data['image']
    img_bytes = img_dict['bytes']
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Create crops folder
    output_dir = "dl1_crops"
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, "dl1_full.jpg"))
    print(f"Full image saved at {os.path.join(output_dir, 'dl1_full.jpg')}")
    
    # Process annotations
    annos = dl1_data['annos']
    print(f"Number of annotations: {len(annos)}")
    
    # Save the prompt for reference
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(dl1_data['prompt'])
    
    # We want to match the user's provided objects if possible, but let's just use the labels from data.
    # The user provided a prompt: "A low-angle cinematic shot of a city skyline, biker man, helmet, biker, couple, man, highway road, city view, bicycle cycle, streetlight, trees, sunset, buildings, sky, road markings, shadows, cyclists, sportswear, sunglasses, socks, shoes, gloves."
    
    for i, anno in enumerate(annos):
        label = anno['category_name'].replace(" ", "_")
        bbox = anno['bbox'] # [x1, y1, x2, y2]
        
        # Crop
        x1, y1, x2, y2 = bbox
        # PIL uses [left, top, right, bottom]
        crop = img.crop((x1, y1, x2, y2))
        
        # Save name based on label and index
        crop_name = f"{label}_{i}.jpg"
        crop.save(os.path.join(output_dir, crop_name))
        print(f"Saved {crop_name} for category '{label}'")
else:
    print("dl1 not found in any parquet file.")
