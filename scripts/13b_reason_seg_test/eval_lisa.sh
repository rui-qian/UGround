#!/bin/bash

echo "=== Unified Visual Grounding Evaluation Framework ==="

# Get script directory and source the dataset selector
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

# Interactive dataset selection
MODEL_KEY="LISA"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

echo ""
echo "Starting $MODEL_KEY evaluation..."
echo "==============================="


# Auto-select available port
source "$(dirname "${BASH_SOURCE[0]}")/port_selector.sh"

deepspeed --include "localhost:0" --master_port="$MASTER_PORT" train_ds.py \
  --model_key="$MODEL_KEY" \
  --version="../s2p-new-13b.bak/runs/sesame_reasoning_test/hg_model_reasonseg_test" \
  --dataset_dir='../dataset_sesame' \
  --dataset="vqa" \
  --sample_rates="1" \
  --exp_name="LISA-7b/eval" \
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
  --no_resume

echo ""
echo "$MODEL_KEY evaluation completed!" 