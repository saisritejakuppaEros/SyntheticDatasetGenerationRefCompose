export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/pose_model"  # your save path
export CONFIG="./default_config.yaml"
export TRAIN_DATA="./examples/pose.jsonl" # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "A girl in the city." \
    --num_train_epochs=1000 \
    --validation_steps=20 \
    --checkpointing_steps=20 \
    --spatial_test_images "./examples/openpose_data/1.png" \
    --subject_test_images None \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=2
