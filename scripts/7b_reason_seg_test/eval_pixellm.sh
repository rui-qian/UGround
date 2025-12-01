#!/bin/bash

echo "=== Unified Visual Grounding Evaluation Framework ==="

# Get script directory and source the dataset selector
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

# Interactive dataset selection
MODEL_KEY="PixelLM"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

echo ""
echo "Starting $MODEL_KEY evaluation..."
echo "================================="

# Auto-select available port
source "$(dirname "${BASH_SOURCE[0]}")/port_selector.sh"

deepspeed --include "localhost:5" --master_port="$MASTER_PORT" train_ds.py \
  --model_key="$MODEL_KEY" \
  --version="../dataset_sesame/PixelLM-13B/hf_model" \
  --dataset_dir='../dataset_sesame' \
  --dataset="vqa" \
  --sample_rates="1" \
  --exp_name="PixelLM-13b/eval" \
  --vision_tower='../dataset_sesame/clip-vit-large-patch14-336' \
  --seg_token_num=3 \
  --num_classes_per_question=1 \
  --batch_size=1 \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_448.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=2 \
  --separate_mm_projector \
  --eval_only \
  --val_dataset="$VAL_DATASET" \
  --no_resume

echo ""
echo "$MODEL_KEY evaluation completed!"
