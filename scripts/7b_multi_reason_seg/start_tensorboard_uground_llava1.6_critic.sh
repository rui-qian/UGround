#!/bin/bash

# TensorBoard startup script
# For viewing UGround training logs

LOG_DIR="./runs/UGround-7b_reason_seg_val_llava1.6_critic"
HOST="0.0.0.0"  # Allow external access
PORT="6006"     # Default port

echo "Starting TensorBoard service..."
echo "Log directory: $LOG_DIR"
echo "Local access: http://localhost:$PORT"
echo "External access: http://$(ip route get 8.8.8.8 | grep -oP 'src \K[\d.]+'):$PORT"
echo ""


# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist!"
    echo "Available log directories:"
    ls -la runs/ 2>/dev/null || echo "   runs/ directory does not exist"
    exit 1
fi

# Check if log directory has content
if [ -z "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
    echo "Warning: Log directory '$LOG_DIR' is empty!"
    echo "Please ensure training has started and generated log files"
fi

echo "Launching TensorBoard..."
tensorboard --logdir="$LOG_DIR" --host="$HOST" --port="$PORT" --reload_interval=30

# If tensorboard command fails
if [ $? -ne 0 ]; then
    echo ""
    echo "TensorBoard startup failed!"
    echo "Please check:"
    echo "  1. Is tensorboard installed: pip install tensorboard"
    echo "  2. Is port $PORT already in use"
    echo "  3. Is the log directory path correct"
    exit 1
fi 