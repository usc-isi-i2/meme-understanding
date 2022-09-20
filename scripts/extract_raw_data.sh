#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SOURCE_PATH=$1

EXTRACTED_DATA_DIRECTORY=data/extracted

echo Deleting older extracted files at $EXTRACTED_DATA_DIRECTORY
rm -rf data/extracted

echo "Extracting raw data files"
mkdir -p $EXTRACTED_DATA_DIRECTORY
unzip $SOURCE_PATH/training.zip -d $EXTRACTED_DATA_DIRECTORY > logs/mami_training_extraction.log
unzip -P *MaMiSemEval2022! $SOURCE_PATH/test.zip -d $EXTRACTED_DATA_DIRECTORY > logs/mami_test_extraction.log

cp $SOURCE_PATH/test_labels.txt data/extracted/test_labels.txt
