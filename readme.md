Task

1. Prepare Synthetic Dataset For training the models

Main goals is to
1. generate things in scale, rotation
2. change the lightning conditions
3. interactions with the objects

So you have landmarks, objects, cusine, star people.

for celeb: /mnt/data0/teja/research_multiref/RefCompose/benchmarking/celeb/CelebA-HQ-img
for objects: /mnt/data0/teja/research_multiref/cultural_dataset/Stanford_Online_Products/*/*
for cusine: /mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images/*/cuisine
for landmarks: /mnt/data0/teja/research_multiref/cultural_dataset/cube_t2i/dataset/images/*/landscapes



vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --served-model-name qwen2.5-vl \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --port 8000

vllm==0.11


1. caption the images 
2. once captinoed the image - make an llm to generate them cinematic captions
3. generate the image
4. once generated image verify them
5. make dift to make canvas
6. get the depth

