#!/bin/bash

echo "=== Unified Visual Grounding Training Framework ==="

MODEL_KEY="PixelLM"

# Default PixelLM training
echo "Training with $MODEL_KEY model..."
# Auto-select available port
source "$(dirname "${BASH_SOURCE[0]}")/port_selector.sh"

deepspeed --master_port="$MASTER_PORT" --include "localhost:7" train_ds.py \
  --model_key="$MODEL_KEY" \
  --version="../dataset_sesame/llava-v1.6-vicuna-7b" \
  --dataset_dir='../dataset_sesame' \
  --exp_name="PixelLM-7b" \
  --vision_tower='../dataset_sesame/clip-vit-large-patch14-336' \
  --seg_token_num=3 \
  --num_classes_per_question=3 \
  --batch_size=2 \
  --grad_accumulation_steps=10 \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_448.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=2 \
  --separate_mm_projector \
  --resume_from_best \
  --lr=0.0003 \
  --dice_loss_weight=4 \
  --dataset="sem_seg||refer_seg||neg_refer_seg||correct_refer_seg||vqa||reason_seg||reason_seg_plus||multi_reason_seg" \
  --sample_rates="12,12,0,0,3,0,0,3" \
  --steps_per_epoch=500 \
  --epochs=30 \
  # --no_resume

echo "$MODEL_KEY training completed!"

# Display available models
echo "Available models:"
python -c "
from model.model_factory import get_available_models
models = get_available_models()
for key, name in models.items():
    print(f'  {key}: {name}')
" 
