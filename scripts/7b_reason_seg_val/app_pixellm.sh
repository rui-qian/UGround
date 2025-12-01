#!/bin/bash

# Visual Grounding Launcher
# Usage: ./app.sh [model_key] [additional_args...]
# Example: ./app.sh PixelLM --precision=fp16

export CUDA_VISIBLE_DEVICES=6
MODEL_KEY=${1:-"PixelLM"} 

if [[ "$1" =~ ^[a-zA-Z] ]] && [[ "$1" != --* ]]; then
    shift  
fi

echo "Starting Visual Grounding with Model: $MODEL_KEY"
echo "Available models: $MODEL_KEY"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo ""

python3 app.py \
	--model_key="$MODEL_KEY" \
	--version="../dataset_sesame/PixelLM-7B/hf_model" \
	--precision='bf16' \
	--seg_token_num=3 \
	--pad_train_clip_images \
	--preprocessor_config='./configs/preprocessor_448.json' \
	--resize_vision_tower \
	--resize_vision_tower_size=448 \
	--vision-tower='../dataset_sesame/clip-vit-large-patch14' \
	--vision_tower_for_mask \
	--image_feature_scale_num=2 \
	--separate_mm_projector \
	"$@"

echo ""
echo "Visual Grounding stopped."
