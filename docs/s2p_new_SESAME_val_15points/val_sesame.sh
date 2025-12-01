##--version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
#        #--version="./hg_model" \
#deepspeed --master_port=24996 --include "localhost:0" train_sesame.py \
#        --version="./hg_model_fprefcoco" \
#        --dataset_dir='../dataset_sesame' \
#        --vision_pretrained="../dataset_sesame/sam_vit_h_4b8939.pth" \
#        --exp_name="sesame_reasoning_fprefcoco" \
#        --eval_only \
#        --dataset="refer_seg" \
#        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
#        --model_max_length=2048
#--version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
        #--version="./hg_model" \
deepspeed --master_port=24997 --include "localhost:5" train_sesame.py \
        --version="runs/sesame_reasoning_val_15/sesame_reasoning_val_15" \
        --dataset_dir='../dataset_sesame' \
        --vision_pretrained="../dataset_sesame/sam_vit_h_4b8939.pth" \
        --exp_name="sesame_reasoning_test" \
        --eval_only \
        --dataset="reason_seg" \
        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
        --model_max_length=2048
