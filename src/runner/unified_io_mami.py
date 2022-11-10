from argparse import ArgumentParser

from PIL import Image
import numpy as np

from src.configs.config_reader import read_json_configs
from src.datasets.mami import MisogynyDataset
from src.models.uio.runner import ModelRunner


arg_parser = ArgumentParser()
arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
arg_parser.add_argument('--configs', required=True, help='configs file from src/configs directory')
args = arg_parser.parse_args()


uio = ModelRunner("base", "./src/models/unified-io-inference/base.bin")

configs = read_json_configs(args.configs)
train_dataset = MisogynyDataset(configs, './data/extracted/TRAINING', 'training.csv')
test_dataset = MisogynyDataset(configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

image_path = test_dataset[0]['input']['image']
print(image_path)
image = np.array(Image.open(image_path))
out = uio.caption(image)
print(out)
