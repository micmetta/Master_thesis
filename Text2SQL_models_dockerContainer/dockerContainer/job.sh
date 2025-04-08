#!/bin/bash
echo "Received question: $1"
echo "Session ID: $2"
echo "PATH_BASE_MODEL_FOR_INFERENCE: $3"
echo "PATH_LoRA1_FOR_INFERENCE: $4"
echo "PATH_LoRA2_FOR_INFERENCE: $5"
echo "Model type: $6"

if [ "$6" = "model8B" ]; then
    python /app/script/inference_8B.py --question "$1" --session_id "$2" --PATH_BASE_MODEL_FOR_INFERENCE "$3" --PATH_LoRA1_FOR_INFERENCE "$4" --PATH_LoRA2_FOR_INFERENCE "$5"
elif [ "$6" = "model70B" ]; then
    python /app/script/inference_70B.py --question "$1" --session_id "$2" --PATH_BASE_MODEL_FOR_INFERENCE "$3" --PATH_LoRA1_FOR_INFERENCE "$4" --PATH_LoRA2_FOR_INFERENCE "$5"
else
    echo "Unknown model type: $6"
    exit 1
fi
