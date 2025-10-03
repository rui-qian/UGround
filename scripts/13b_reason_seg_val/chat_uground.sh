export CUDA_VISIBLE_DEVICES=7

python3 chat.py \
    --version="runs/UGround-7b_reason_seg_val/hf-UGround-7b_reason_seg_val" \
	--model_key="UGround" \
	--precision='bf16' \
	--seg_token_num=1  \
	--vision-tower='../dataset_sesame/clip-vit-large-patch14-336' \
	--image_feature_scale_num=1 \
	--pad_train_clip_images \
	--preprocessor_config='./configs/preprocessor_336.json' \
	--separate_mm_projector \
	# --resize_vision_tower \
	# --resize_vision_tower_size=336 \
	# --vision_tower_for_mask \

