#!/bin/bash
echo "Received question: $1"
echo "Session ID: $2"
echo "PATH_BASE_MODEL_REMOTE_FOR_INFERENCE: $3"
echo "PATH_MODEL_USE_REMOTE_FOR_INFERENCE: $4"
echo "Model type: $5"

if [ "$5" = "model8B" ]; then
    python /app/script/inference_8B.py --question "$1" --session_id "$2" --PATH_BASE_MODEL_REMOTE_FOR_INFERENCE "$3" --PATH_MODEL_USE_REMOTE_FOR_INFERENCE "$4"
elif [ "$5" = "model70B" ]; then
    python /app/script/inference_70B.py --question "$1" --session_id "$2" --PATH_BASE_MODEL_REMOTE_FOR_INFERENCE "$3" --PATH_MODEL_USE_REMOTE_FOR_INFERENCE "$4"
else
    echo "Unknown model type: $5"
    exit 1
fi
