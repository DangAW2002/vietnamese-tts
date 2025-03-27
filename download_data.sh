#!/bin/bash

# Make sure the script exits on any error
set -e

echo "Installing required packages for downloading data..."
pip install gdown

echo "Downloading and extracting data..."
python src/utils/download_data.py

echo "Data download and extraction completed successfully!"