export CUDA_VISIBLE_DEVICES=7 
python tools/sesame_demo.py \
	--version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
	--vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
	--model_max_length=2048
