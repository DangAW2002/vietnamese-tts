#!/bin/bash

# Make sure the script exits on any error
set -e

echo "Starting model training process..."

# Check if data exists, if not download it
if [ ! -d "data" ]; then
    echo "Data directory not found, running data download script first..."
    ./download_data.sh
fi

# Run the training script with Python 3
echo "Starting training process..."
python3 src/train.py --batch_size 8 --pad_to_fixed_length True --max_seq_len 128

echo "Training completed! You can now use the trained model with run_model.sh"
echo "Example usage: ./run_model.sh checkpoints/best_model.pt"