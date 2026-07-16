hari om, om namah shivaya

## Indian heritage interior — FLUX.2 inference

Built-in prompt (also in `flux2_heritage_infer.py`):

> Design a luxurious Indian heritage-inspired interior space with the following traditional decor elements integrated naturally and aesthetically: a large Brass Urli filled with floating flowers and candles, an ornate carved Jharokha wall frame, an intricate Jaali Partition casting patterned shadows, a vibrant Madhubani Painting on the wall, a rich Tanjore Painting featuring traditional motifs, a tall Antique Brass Lamp placed near the seating area, a carved wooden swing suspended elegantly in the room, a handcrafted Bidriware Vase with silver inlay detailing, colorful hanging Kathputli Puppets as accent decor, and a handwoven Dhurrie rug with geometric ethnic patterns. The interior should blend traditional Indian craftsmanship with modern luxury styling, featuring warm ambient lighting, earthy textures, teakwood finishes, sandstone accents, soft sunlight entering through carved screens, cinematic composition, ultra-detailed decor styling, photorealistic materials, premium interior photography aesthetic, balanced color palette, high realism, 8k detail, architectural digest style, cozy yet royal atmosphere.

### Quick start

```bash
cd /mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer_v2

# 4 variants (default), seeds 42–45
CUDA_VISIBLE_DEVICES=0 ./run.sh

# Single image
VARIANTS=1 ./run.sh

# Direct Python (more flags)
CUDA_VISIBLE_DEVICES=0 python flux2_heritage_infer.py --variants 4 --seed 42
```

Outputs land in `outputs/` as `heritage_interior_var01_seed42.png`, etc., plus a `.txt` sidecar with the prompt.

### Depth (Depth Anything 3)

For an existing RGB image (default: `generated_image/image_gen.png`):

```bash
# one-time: pip install -r requirements-da3.txt

CUDA_VISIBLE_DEVICES=3 ./run_depth.sh

# or directly
CUDA_VISIBLE_DEVICES=3 python stage2_depth.py
CUDA_VISIBLE_DEVICES=3 python stage2_depth.py -i generated_image/image_gen.png

# batch all images under a folder → <folder>/depth/<stem>_depth.png
CUDA_VISIBLE_DEVICES=3 python stage2_depth.py -i generated_image --batch
CUDA_VISIBLE_DEVICES=3 python stage2_depth.py -i outputs --batch
```

Writes `generated_image/image_gen_depth.png` (16-bit grayscale `I;16`, training-compatible).

### LoRA (canvas + depth conditioning)

Uses `generated_image/black_canvas.jpg`, `generated_image/image_gen_depth.png`, and `generated_image/image_gen.txt` by default.

```bash
chmod +x run_lora.sh

# Default: outputs/image_gen_lora.png
CUDA_VISIBLE_DEVICES=0 ./run_lora.sh

# Depth-first for first 4 denoise steps, then appearance-heavy
LORA_FIRST_DEPTH_STEPS=4 CUDA_VISIBLE_DEVICES=0 ./run_lora.sh

# Direct Python
CUDA_VISIBLE_DEVICES=0 python stage3_lora_infer.py --no-canvas_depth_coarse_augment
CUDA_VISIBLE_DEVICES=0 python stage3_lora_infer.py --lora_first_depth_steps 4 --seed 42
```

Override checkpoint via `LORA_PATH` or `--lora_path` (default: `lora_training_v2/.../checkpoint-11000/lora.safetensors`).

### Common options

| Flag / env | Default | Notes |
|------------|---------|-------|
| `--variants` / `VARIANTS` | 4 | Use `--single` or `VARIANTS=1` for one file |
| `--seed` / `SEED` | 42 | Base seed; each variant adds `--seed-step` (default 1) |
| `--num-inference-steps` / `STEPS` | 50 | Denoise steps |
| `--guidance-scale` / `GUIDANCE` | 4.0 | CFG |
| `--width` / `WIDTH` | 1280 | |
| `--height` / `HEIGHT` | 720 | |
| `FLUX2_MODEL` | FLUX.2-dev snapshot | Override base model path |

### Server mode (optional)

If `infer/flux_server/server_flux.py` is already running:

```bash
python ../infer/flux_server/client_flux.py \
  --url http://127.0.0.1:8765 \
  --caption "$(python -c 'from flux2_heritage_infer import HERITAGE_INTERIOR_PROMPT; print(HERITAGE_INTERIOR_PROMPT)')" \
  --variants 4 \
  -o outputs/heritage_interior_server.png
```
