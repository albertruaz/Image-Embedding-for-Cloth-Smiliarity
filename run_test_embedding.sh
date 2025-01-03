#!/bin/bash

# Set which embedding model to use
export EMBEDDING_MODEL_FILE="blip_embedding_model"  # or "clip_embedding_model"
export MODEL_NAME="Salesforce/blip-image-captioning-base"





export SIMILAR_PAIRS='[("theme2-1", "theme2-2", 0.5)]' # ("theme1-1", "theme1-2", 0.1)
export DIFFERENT_PAIRS='[("theme1-1", "theme2-1", 0.5), ("theme1-1", "theme2-2", 0.3)]'
export OTHER_FOLDERS='[]'

# Run the Python script
python test_embedding.py 