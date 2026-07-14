1. generate the highly good images
2. from the generated images generate objects out of it
3. segment them and get a neutral characteristic out of it using flux 2.0 for source image generation. 
4. generate 4 to 8 images dataset so that u have multi reference images. 


1. Fix the objects and generate the prompts
2. generate the images
3. segment the images based on the prompts 
4. gdino the images and then cross verify it.
5. do the inference on them

python objects.py --total 50
python flux2_imagegen.py
python bbox_annotation.py


cd /mnt/data0/teja/research_multiref/synthetic_dataset_generation/v2
source ../.venv/bin/activate


CUDA_VISIBLE_DEVICES=2 python flux2_imagegen.py --shard_id 0 --num_shards 6 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python flux2_imagegen.py --shard_id 1 --num_shards 6 --device cuda:0

CUDA_VISIBLE_DEVICES=4 python flux2_imagegen.py --shard_id 2 --num_shards 6 --device cuda:0

CUDA_VISIBLE_DEVICES=5 python flux2_imagegen.py --shard_id 3 --num_shards 6 --device cuda:0

CUDA_VISIBLE_DEVICES=6 python flux2_imagegen.py --shard_id 4 --num_shards 6 --device cuda:0

CUDA_VISIBLE_DEVICES=7 python flux2_imagegen.py --shard_id 5 --num_shards 6 --device cuda:0