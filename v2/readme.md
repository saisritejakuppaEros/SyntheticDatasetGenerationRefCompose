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