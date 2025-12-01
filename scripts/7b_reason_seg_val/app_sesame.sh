#!/bin/bash

# Visual Grounding Launcher
# Usage: ./app.sh [model_key] [additional_args...]
# Example: ./app.sh PixelLM --precision=fp16

export CUDA_VISIBLE_DEVICES=6
MODEL_KEY=${1:-"SESAME"}  

if [[ "$1" =~ ^[a-zA-Z] ]] && [[ "$1" != --* ]]; then
    shift  
fi

echo "Starting Visual Grounding with Model: $MODEL_KEY"
echo "Available models: $MODEL_KEY"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo ""

python3 app.py \
	--model_key="$MODEL_KEY" \
	--version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
	--precision='bf16' \
	--seg_token_num=1 \
	--vision-tower='../dataset_sesame/clip-vit-large-patch14-336' \
	--image_feature_scale_num=1 \
	"$@"

echo ""
echo "Visual Grounding stopped."
