export CUDA_VISIBLE_DEVICES=7 
#python demo.py --pretrained_model_path="../dataset/SESAME-LLaVA-v1.5-7B"
python demo.py \
	--pretrained_model_path="runs/sesame_reasoning_val_15/sesame_reasoning_val_15" \
	--vision_tower="../dataset_sesame/clip-vit-large-patch14" \
	--model_max_length=2048
