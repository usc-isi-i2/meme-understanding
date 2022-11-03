import json

from tqdm import tqdm

from src.datasets.mami import MisogynyDataset
from src.configs.config_reader import read_json_configs

configs = read_json_configs('src/configs/knn/local.json')

train_dataset = MisogynyDataset(configs, 'data/extracted/TRAINING', 'training.csv')
test_dataset = MisogynyDataset(configs, 'data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')


labels = {}
for sample in tqdm(train_dataset + test_dataset):
    labels[sample['input']['image']] = sample['output']

with open('./data/processed/labels.json', 'w') as f:
    json.dump(labels, f)
