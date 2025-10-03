#!/bin/bash

echo "=== Unified Visual Grounding Evaluation Framework ==="

# Get script directory and source the dataset selector
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

# Interactive dataset selection
MODEL_KEY="READ"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

echo ""
echo "Starting $MODEL_KEY evaluation..."
echo "==============================="


# Auto-select available port
source "$(dirname "${BASH_SOURCE[0]}")/port_selector.sh"

deepspeed --include "localhost:5" --master_port="$MASTER_PORT" train_ds.py \
  --model_key="$MODEL_KEY" \
  --version="../s2p-new-SESAME_val_15points/runs/sesame_reasoning_val_15/sesame_reasoning_val_15/" \
  --dataset_dir='../dataset_sesame' \
  --dataset="vqa" \
  --sample_rates="1" \
  --exp_name="READ-7b/eval" \
  --vision_tower='../dataset_sesame/clip-vit-large-patch14-336' \
  --vision_pretrained="../dataset_sesame/sam_vit_h_4b8939.pth" \
  --seg_token_num=1 \
  --num_classes_per_question=3 \
  --batch_size=1 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=1 \
  --eval_only \
  --val_dataset="$VAL_DATASET" \
  --no_resume \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_336.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=336 \
  --separate_mm_projector \

echo ""
echo "$MODEL_KEY evaluation completed!" 
