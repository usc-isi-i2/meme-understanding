from argparse import ArgumentParser

from PIL import Image
import numpy as np
from tqdm import tqdm

from src.configs.config_reader import read_json_configs
from src.datasets.meme import MemeDataset
from src.models.uio.runner import ModelRunner


arg_parser = ArgumentParser()
arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
arg_parser.add_argument('--configs', required=True, help='configs file from src/configs directory')
args = arg_parser.parse_args()


uio = ModelRunner("large", "large.bin")

configs = read_json_configs(args.configs)
train_dataset = MemeDataset.create_mami_dataset_from_files('train', configs, './data/extracted/TRAINING', 'training.csv')
test_dataset = MemeDataset.create_mami_dataset_from_files('test', configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

for sample in tqdm(test_dataset):
  image_path = test_dataset[0]['input']['image']
  image = np.array(Image.open(image_path))
  classification = uio.image_classification(image, answer_options=["misogynous", "not misogynous"])


