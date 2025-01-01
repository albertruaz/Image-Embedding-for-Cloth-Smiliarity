#!/bin/bash

# Set environment variables
export MODEL_NAME="openai/clip-vit-base-patch32"
export DATA_DIR="./data"
export SIMILAR_PAIRS='[("theme1-1", "theme1-2", 1.0), ("theme2-1", "theme2-2", 0.8)]'
export DIFFERENT_PAIRS='[("theme1-1", "theme2-1", 1.0), ("theme1-1", "theme2-2", 0.5)]'
export OTHER_FOLDERS='[]'

# Run the Python script
python test_embedding.py 