#!/bin/bash

# Dataset mapping
declare -a DATASETS=(
    "ReasonSeg|val"
    "ReasonSeg|test_overall"
    "ReasonSeg|test_shortquery"
    "ReasonSeg|test_longquery"
    "refcocog|umd|val"
    "refcocog|umd|test"
    "refcoco+|unc|val"
    "refcoco+|unc|testA"
    "refcoco+|unc|testB"
    "refcoco|unc|val"
    "refcoco|unc|testA"
    "refcoco|unc|testB"
    "multirefcocog|umd|val"
    "multirefcocog|umd|test"
    "multirefcoco+|unc|val"
    "multirefcoco+|unc|testA"
    "multirefcoco+|unc|testB"
    "multirefcoco|unc|val"
    "multirefcoco|unc|testA"
    "multirefcoco|unc|testB"
    "MultiReasonSeg|val"
    "MultiReasonSeg|test"
    "MultiReasonSeg|test_less"
    "MultiReasonSeg|test_many"
    "grefcoco|unc|val"
    "grefcoco|unc|testA"
    "grefcoco|unc|testB"
    "refzom|final|test"
    "fprefcoco|val"
    "fprefcoco+|val"
    "fprefcocog|val"
    "fpReasonSeg|val"
    "fpReasonSeg|test"
)

# Function to show available datasets
show_datasets() {
    echo ""
    echo "Available datasets for evaluation:"
    echo "======================================"
    for i in "${!DATASETS[@]}"; do
        printf "%2d: %s\n" "$i" "${DATASETS[$i]}"
    done
    echo ""
}

# Function to build val_dataset string
build_val_dataset() {
    local indices="$1"
    local val_dataset=""
    
    # Parse range (e.g., "0-2" or "0,2,4" or "0-2,5,7-9")
    IFS=',' read -ra PARTS <<< "$indices"
    local selected=()
    
    for part in "${PARTS[@]}"; do
        if [[ $part == *"-"* ]]; then
            # Handle range (e.g., "0-2")
            IFS='-' read -ra RANGE <<< "$part"
            start=${RANGE[0]}
            end=${RANGE[1]}
            for ((i=start; i<=end; i++)); do
                if [[ $i -ge 0 && $i -lt ${#DATASETS[@]} ]]; then
                    selected+=($i)
                fi
            done
        else
            # Handle single number
            if [[ $part -ge 0 && $part -lt ${#DATASETS[@]} ]]; then
                selected+=($part)
            fi
        fi
    done
    
    # Remove duplicates and sort
    selected=($(printf "%s\n" "${selected[@]}" | sort -nu))
    
    # Build val_dataset string
    for i in "${selected[@]}"; do
        if [[ -n $val_dataset ]]; then
            val_dataset="${val_dataset}||${DATASETS[$i]}"
        else
            val_dataset="${DATASETS[$i]}"
        fi
    done
    
    echo "$val_dataset"
}

# Interactive dataset selection function
select_datasets_interactive() {
    local model_name="$1"
    
    echo "$model_name Evaluation - Dataset Selection" >&2
    echo "========================================" >&2
    
    # Show available datasets to stderr so they appear on screen
    show_datasets >&2
    
    # Get user input
    echo "   Input formats:" >&2
    echo "   Single:    0" >&2
    echo "   Range:     0-2" >&2
    echo "   Multiple:  0,2,4" >&2
    echo "   Mixed:     0-2,5,8-10" >&2
    echo "" >&2
    
    while true; do
        read -p "Enter dataset indices to evaluate: " INDICES
        
        if [[ -z "$INDICES" ]]; then
            echo "Please enter valid dataset indices!" >&2
            continue
        fi
        
        # Build val_dataset parameter
        VAL_DATASET=$(build_val_dataset "$INDICES")
        
        if [[ -z "$VAL_DATASET" ]]; then
            echo "No valid datasets selected! Please try again." >&2
            continue
        fi
        
        echo "" >&2
        echo "Selected datasets: $VAL_DATASET" >&2
        echo "" >&2
        read -p "Start evaluation with these datasets? (y/n): " CONFIRM
        
        if [[ $CONFIRM =~ ^[Yy]$ ]]; then
            break
        else
            echo "Please select datasets again..." >&2
            echo "" >&2
        fi
    done
    
    # Return the selected dataset string to stdout
    echo "$VAL_DATASET"
} 