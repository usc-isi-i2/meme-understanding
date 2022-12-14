# meme-understanding

## Raw datasets
Link for the raw datasets:
* MAMI: https://competitions.codalab.org/competitions/34175#learn_the_details
* Hateful memes: https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/

## Setting up environment
* [M1](./environments/m1.md) 
* [x86 Linux](./environments/linux.md)
* [Colab](./environments/colab.md)

## To extract data
Run following scripts from repo root directory to extract data into `./data/extracted` for MAMI and `./data/extracted_hateful_meme` for hateful meme dataset

* `sh scripts/extract_mami_data.sh <raw_data_path>`
* `sh scripts/extract_hateful_meme.sh <path/hateful_memes.zip>`

## This repo supports
* Meme classification using neural network `src/runner/classify_mami.py`
* Meme classification using xDNN `src/runner/classify_xdnn.py`
* Meme classification and similar feature extraction `src/runner/mami_knn_clip_runner.py`

## Experiments
The experiments were ran through [vscode launch config](.vscode/launch.json), please refer the same as the execution instructions.
