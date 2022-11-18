# meme-understanding

# Raw dataset

# Setting up environment
[M1](./environments/m1.md) [Colab](./environments/colab.md)

# To extract data
`sh scripts/extract_mami_data.sh <raw_data_path>`
`sh scripts/extract_hateful_meme.sh <path/hateful_memes.zip>`

# Runs
* knn clip classification: 
* Training Linear classifier over clip features

## Mami

### Linear Classification
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=bert.json --device=cuda:1 > ./logs/bertweet.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=bertweet.json --device=cuda:1 > ./logs/bertweet.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=clip.json --device=cuda:2 > ./logs/clip.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=combined.json --device=cuda:3 > ./logs/combined.log &`

### KNN classification
`PYTHONPATH=. python src/runner/mami_knn_clip_runner.py --device=mps`

## Hateful Memes
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=hateful-bert.json --device=cuda:1 > ./logs/hateful-bertweet.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=hateful-bertweet.json --device=cuda:1 > ./logs/hateful-bertweet.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=hateful-clip.json --device=cuda:2 > ./logs/hateful-clip.log &`
`PYTHONPATH=. nohup python src/runner/classify_mami.py --config=hateful-combined.json --device=cuda:3 > ./logs/hateful-combined.log &`
