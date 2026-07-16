# RefCompose

Official codebase for **reference-guided image composition** with FLUX diffusion models — canvas layout + depth conditioning, spatial/subject/style control, and LoRA fine-tuning.

## Documentation

Full documentation lives in the [`docs/`](docs/) folder:

- [Overview & index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Dataset preparation](docs/dataset-preparation.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [API reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)

## Quick start

```bash
# Inference (requires trained LoRA + FLUX.2 weights)
cd scripts/infer_v2
CUDA_VISIBLE_DEVICES=0 ./run.sh          # text-to-image
CUDA_VISIBLE_DEVICES=0 ./run_depth.sh    # depth estimation
CUDA_VISIBLE_DEVICES=0 ./run_lora.sh     # canvas + depth LoRA refine

# Training (requires prepared dataset — see docs)
cd scripts/train
CANVAS_DATA_ROOTS=/path/to/output ./train_flux2_lora.sh
```

# get the models 
```
cd ImageComposition
hf download saisriteja/ImageComposition \
    --repo-type dataset \
    --local-dir ./models
```

# Setting Up the env
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1

uv pip install \
diffusers==0.32.2 \
easydict==1.13 \
einops==0.8.1 \
peft==0.14.0 \
pillow==11.0.0 \
protobuf==5.29.3 \
requests==2.32.3 \
safetensors==0.5.2 \
sentencepiece==0.2.0 \
spaces==0.34.1 \
transformers==4.49.0 \
datasets \
wandb \
ultralytics
```



#benchmarking
mkdir benchmarking
hf download multimedia-synergy-lab/CogCanvas \
    --repo-type dataset \
    --local-dir .
