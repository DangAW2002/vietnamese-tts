#!/bin/bash

# Make sure the script exits on any error
set -e

# Check if a checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the path to the trained model checkpoint."
    echo "Usage: ./run_model.sh path/to/checkpoint"
    exit 1
fi

# Set the checkpoint path from the first argument
CHECKPOINT="$1"

echo "Running TTS model with checkpoint: $CHECKPOINT"

# Run the model inference (modify this command based on your actual inference script)
python src/models/inference.py --checkpoint "$CHECKPOINT"

echo "Model execution completed."