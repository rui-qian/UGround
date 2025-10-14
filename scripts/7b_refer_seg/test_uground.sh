#!/bin/bash

function run_inference() {
    MODEL_KEY="${1}"
    CUDA_DEVICE="${2}"
    PROCESS_NUM="${3}"
    WORLD_SIZE="${4}"
    DATASET="${5}"
    INFERENCE_CMD="${6:-inference}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python test_ds.py \
        --model_key="${MODEL_KEY}" \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir ../dataset_sesame \
	    --version="runs/UGround-7b_reason_seg_val_llava1.5/hf-UGround-7b_reason_seg_val_llava1.5" \
        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
        --separate_mm_projector \
        --pad_train_clip_images \
        --preprocessor_config='./configs/preprocessor_336.json' \
        --resize_vision_tower \
        --resize_vision_tower_size=336 \
        --vision_tower_for_mask \
        --model_max_length=2048 \
        --val_dataset="${DATASET}" \
        --vis_save_path="./inference_results/${DATASET}_inference_cvpr" \
        --num_layers=33 \
        --strategy="random_walker" \
        --mode=0 \
        --temperature=1.0 \
        --hard \
        --num_heads=3 \
        --eval_legacy 
}

# Get script directory and source the dataset selector
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

# Interactive dataset selection
MODEL_KEY="UGround"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

# Split the selected datasets
IFS=$'\n' read -rd '' -a SELECTED_DATASETS <<< "${VAL_DATASET//||/$'\n'}"
# declare -a datasets=("fprefcoco|val" "fprefcoco+|val" "fprefcocog|val" "refcoco|val" "refcoco+|val" "refcocog|val")
# for dataset in "${datasets[@]}"; do
for dataset in "${SELECTED_DATASETS[@]}"; do
    echo "Running inference for ${dataset}..."
    run_inference "${MODEL_KEY}" 5 0 1 "${dataset}" "inference"
    echo "Waiting for background inference processes to finish... for ${dataset}..."
    wait
    echo "Background processes for ${dataset} finished. Running metrics..."
    run_inference "${MODEL_KEY}" 5 0 1 "${dataset}" "metrics" 
    echo "Inference and metrics for ${dataset} finished."
done
