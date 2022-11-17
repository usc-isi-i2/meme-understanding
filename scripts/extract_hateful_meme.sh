#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SOURCE_PATH=$1

EXTRACTED_DATA_DIRECTORY=data/extracted_hateful_meme
PROCESSED_DATA_DIRECTORY=data/processed_hateful_meme

echo Deleting older extracted files at $EXTRACTED_DATA_DIRECTORY
rm -rf data/extracted_hateful_meme

echo "Extracting raw data files"
mkdir -p $EXTRACTED_DATA_DIRECTORY
unzip $SOURCE_PATH -d $EXTRACTED_DATA_DIRECTORY > logs/mami_training_extraction.log

echo "Creating directory for processed data files"
mkdir -p $PROCESSED_DATA_DIRECTORY
