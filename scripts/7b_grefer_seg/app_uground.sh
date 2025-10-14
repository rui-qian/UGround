#!/bin/bash

# Visual Grounding Launcher
# Usage: ./app.sh [model_key] [additional_args...]
# Example: ./app.sh PixelLM --precision=fp16

export CUDA_VISIBLE_DEVICES=7
MODEL_KEY=${1:-"UGround"}  

if [[ "$1" =~ ^[a-zA-Z] ]] && [[ "$1" != --* ]]; then
    shift  
fi

echo "Starting Visual Grounding with Model: $MODEL_KEY"
echo "Available models: $MODEL_KEY"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo ""

python3 app.py \
	--model_key="$MODEL_KEY" \
	--version="runs/UGround-7b_reason_seg_val_llava1.5/hf-UGround-7b_reason_seg_val_llava1.5" \
	--precision='bf16' \
	--seg_token_num=1 \
	--vision-tower='../dataset_sesame/clip-vit-large-patch14-336' \
	--image_feature_scale_num=1 \
	--pad_train_clip_images \
	--preprocessor_config='./configs/preprocessor_336.json' \
	--separate_mm_projector \
	"$@"

echo ""
echo "Visual Grounding stopped."
