# python caption_images.py --limit-per-category 20
# python llm_synthetic_captions.py --num-samples 100000

# export CUDA_VISIBLE_DEVICES=1


# export NCCL_NVLS_ENABLE=0
# export NCCL_TREE_THRESHOLD=0
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_P2P_LEVEL=SYS
# export NCCL_SHM_DISABLE=0
# export NCCL_ALGO=Ring
# export NCCL_TIMEOUT=1800
# export NCCL_DEBUG=WARN
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager

cd /mnt/data0/teja/research_multiref/synthetic_dataset_generation
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 0 --num_shards 6 --device "cuda:0" \
    --no_skip_existing --limit 5 --save_collage_debug


CUDA_VISIBLE_DEVICES=1 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 1 --num_shards 6 --device "cuda:0" 

CUDA_VISIBLE_DEVICES=2 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 2 --num_shards 6 --device "cuda:0" 

CUDA_VISIBLE_DEVICES=3 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 3 --num_shards 6 --device "cuda:0" 

CUDA_VISIBLE_DEVICES=4 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 4 --num_shards 6 --device "cuda:0" 

CUDA_VISIBLE_DEVICES=5 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 5 --num_shards 6 --device "cuda:0" 

CUDA_VISIBLE_DEVICES=6 python flux2_gen.py \
    --manifest "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage2_composed/manifest.jsonl" \
    --images_root "/mnt/data0/teja/research_multiref/synthetic_dataset_generation" \
    --output_dir "/mnt/data0/teja/research_multiref/synthetic_dataset_generation/outputs/stage3_generated" \
    --shard_id 6 --num_shards 6 --device "cuda:0" 