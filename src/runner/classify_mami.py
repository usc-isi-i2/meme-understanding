import os
from argparse import ArgumentParser

import torch
import random

from src.configs.config_reader import read_json_configs
from src.models.utils import get_classification_model
from src.datasets.utils import get_train_dataset, get_test_dataset
from src.logger import Logger
from src.trainer.mami_trainer import MamiTrainer


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='ckg11.json', required=True, help='Config file from src/configs/classifier')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./src/configs/classifier', args.config))
    logger =Logger(configs)

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    train_dataset = get_train_dataset(configs)
    test_dataset = get_test_dataset(configs)
    
    trainer = MamiTrainer(get_classification_model, configs, train_dataset, test_dataset, args.device, logger)
    trainer.train_kfold()
