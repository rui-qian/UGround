#!/bin/bash

# ==================== Configuration ====================
# Experiment directory (should contain ckpt_model/ subdirectory)
EXP_DIRECTORY="./runs/UGround-13b_reason_seg_val_llava1.5_ema"

# Model identifier
MODEL_KEY="UGround"

# Base model path
VERSION="../dataset_sesame/llava-v1.5-13b"

# Vision tower path
VISION_TOWER="../dataset_sesame/clip-vit-large-patch14-336"

# Output directory name (created under EXP_DIRECTORY)
# Leave empty to auto-generate as "hf-{experiment_name}"
OUTPUT_DIR=""

# GPU device to use
GPU_ID="5"
# ======================================================

# Auto-generate output directory name if not specified
if [ -z "$OUTPUT_DIR" ]; then
    # Extract experiment name from EXP_DIRECTORY (last part of path)
    EXP_NAME=$(basename "$EXP_DIRECTORY")
    OUTPUT_DIR="hf-${EXP_NAME}"
fi

# Check if we're in the correct directory
if [ ! -f "merge_lora_weights_and_save_hf_model.py" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    echo "Expected to find: merge_lora_weights_and_save_hf_model.py"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Set paths
HF_CKPT_PATH="${EXP_DIRECTORY}/${OUTPUT_DIR}"

# Print configuration
echo ""
echo "🔧 Merge Configuration:"
echo "   Experiment: $EXP_DIRECTORY"
echo "   Model Key:  $MODEL_KEY"
echo "   Base Model: $VERSION"
echo "   Vision:     $VISION_TOWER"
echo "   Output:     $HF_CKPT_PATH"
echo "   GPU:        $GPU_ID"
echo ""

# Validate directories
if [ ! -d "$EXP_DIRECTORY" ]; then
    echo "Error: Experiment directory not found: $EXP_DIRECTORY"
    exit 1
fi

if [ ! -d "${EXP_DIRECTORY}/best_ckpt_model" ]; then
    echo "Error: Checkpoint directory not found: ${EXP_DIRECTORY}/best_ckpt_model"
    exit 1
fi

if [ ! -d "$VERSION" ]; then
    echo "Error: Base model not found: $VERSION"
    exit 1
fi

# Check if output exists
if [ -d "$HF_CKPT_PATH" ]; then
    echo "Output directory exists: $HF_CKPT_PATH"
    read -p "Do you want to overwrite it? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
    rm -rf "$HF_CKPT_PATH"
fi

# Save current directory
ORIGINAL_DIR=$(pwd)

# Step 1: Convert DeepSpeed checkpoint
echo "Converting DeepSpeed checkpoint..."
TMP_FILE="$(realpath "${EXP_DIRECTORY}/tmp_merge_$(date +%s).bin")"

cd "${EXP_DIRECTORY}/best_ckpt_model"
python zero_to_fp32.py . "$TMP_FILE"

if [ $? -ne 0 ]; then
    echo "DeepSpeed conversion failed"
    cd "$ORIGINAL_DIR"
    exit 1
fi

echo "Checkpoint converted: $TMP_FILE"
cd "$ORIGINAL_DIR"

# Step 2: Merge LoRA weights
echo "Merging LoRA weights..."
python3 merge_lora_weights_and_save_hf_model.py \
  --model_key="$MODEL_KEY" \
  --version="$VERSION" \
  --weight="$TMP_FILE" \
  --save_path="./${HF_CKPT_PATH}" \
  --vision-tower="$VISION_TOWER" \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_336.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=336 \
  --vision_tower_for_mask \
  --separate_mm_projector \
  --lora_r=64 \
  --num_layers=41 \
  --strategy="policy_walker" \
  --mode=3 \
  --baseline_type="ema" \
  --baseline_beta=1.0 \
  --eval_legacy \

if [ $? -ne 0 ]; then
    echo "LoRA merge failed"
    rm -f "$TMP_FILE"
    exit 1
fi

# Clean up
rm -f "$TMP_FILE"

echo ""
echo "Merge completed successfully!"
echo "Model saved to: $HF_CKPT_PATH"
