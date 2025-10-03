#!/bin/bash

echo "=== Unified Visual Grounding Training Framework ==="

MODEL_KEY="UGround"

# Default GSVA training
echo "Training with $MODEL_KEY model..."
# Auto-select available port
source "$(dirname "${BASH_SOURCE[0]}")/port_selector.sh"

deepspeed --master_port="$MASTER_PORT" --include "localhost:7" train_ds.py \
  --model_key="$MODEL_KEY" \
  --version="../dataset_sesame/llava-v1.6-vicuna-7b" \
  --dataset_dir='../dataset_sesame' \
  --exp_name="UGround-7b_reason_seg_val_llava1.6_ema" \
  --vision_tower='../dataset_sesame/clip-vit-large-patch14-336' \
  --vision_pretrained='../dataset_sesame/sam_vit_h_4b8939.pth' \
  --seg_token_num=1 \
  --num_classes_per_question=3 \
  --batch_size=2 \
  --grad_accumulation_steps=10 \
  --use_expand_question_list \
  --image_feature_scale_num=1 \
  --separate_mm_projector \
  --lr=0.0001 \
  --warmup_min_lr=0 \
  --warmup_num_steps=100 \
  --lora_r=8 \
  --dice_loss_weight=4 \
  --resume_from_best \
  --dataset="reason_seg" \
  --sample_rates="12" \
  --val_dataset="ReasonSeg|val" \
  --steps_per_epoch=100 \
  --eval_interval=10 \
  --epochs=30 \
  --num_layers=33 \
  --strategy="policy_walker" \
  --mode=3 \
  --baseline_type="ema" \
  --baseline_beta=1.0 \
  --eval_legacy \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_336.json' \
  # --resize_vision_tower \
  # --resize_vision_tower_size=336 \
  # --vision_tower_for_mask \

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
