# meme-understanding

# Raw dataset

# Setting up environment
[M1](./environments/m1.md) [Colab](./environments/colab.md)

# To extract data
`sh scripts/extract_raw_data.sh <raw_data_path>`

# Runs
* knn clip classification: `PYTHONPATH=. python src/runner/mami_knn_clip_runner.py --device=mps`
* Training Linear classifier over clip features
