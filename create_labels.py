import json

from tqdm import tqdm

from src.datasets.meme import MemeDataset
from src.configs.config_reader import read_json_configs

configs = read_json_configs('src/configs/knn/local.json')

train_dataset = MemeDataset(configs, 'data/extracted/TRAINING', 'training.csv')
test_dataset = MemeDataset(configs, 'data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')


labels = {}
for sample in tqdm(train_dataset + test_dataset):
    labels[sample['input']['image']] = sample['output']

with open('./data/processed/labels.json', 'w') as f:
    json.dump(labels, f)
