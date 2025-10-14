#!/bin/bash

function run_inference() {
    CUDA_DEVICE="${1}"
    PROCESS_NUM="${2}"
    WORLD_SIZE="${3}"
    DATASET="${4}"
    INFERENCE_CMD="${5:-inference}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python test_sesame.py \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir ../dataset_sesame \
	    --version="../dataset_sesame/SESAME-LLaVA-v1.5-7B" \
        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
        --model_max_length=2048 \
        --val_dataset="${DATASET}" \
        --vis_save_path="./inference_results/${DATASET}_inference_cvpr"
}

# Get script directory and source the dataset selector
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

# Interactive dataset selection
MODEL_KEY="READ"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

# Split the selected datasets
IFS=$'\n' read -rd '' -a SELECTED_DATASETS <<< "${VAL_DATASET//||/$'\n'}"
# declare -a datasets=("fprefcoco|val" "fprefcoco+|val" "fprefcocog|val" "refcoco|val" "refcoco+|val" "refcocog|val")
# for dataset in "${datasets[@]}"; do
for dataset in "${SELECTED_DATASETS[@]}"; do
    echo "Running inference for ${dataset}..."
    run_inference 7 0 1 "${dataset}"
    echo "Waiting for background inference processes to finish... for ${dataset}..."
    wait
    echo "Background processes for ${dataset} finished. Running metrics..."
    run_inference 7 0 1 "${dataset}" "metrics"
    echo "Inference and metrics for ${dataset} finished."
done
