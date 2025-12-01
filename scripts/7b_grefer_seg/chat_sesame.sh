export CUDA_VISIBLE_DEVICES=6

python3 chat.py --version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
	--model_key="SESAME" \
	--precision='bf16' \
	--seg_token_num=1  \
	--vision-tower='../dataset_sesame/clip-vit-large-patch14-336' \
	--image_feature_scale_num=1 
