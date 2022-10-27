import os
from argparse import ArgumentParser

import torch
import random

from src.configs.config_reader import read_json_configs
from src.datasets.mami import MisogynyDataset
from src.logs.file_logger import FileLogger
from src.trainer.mami_trainer import MamiTrainer
from src.models.bertweet_classifier import BertTweetClassifier
from src.models.clip import Clip
from src.models.clip_bertweet_classifier import ClipBertTweetClassifier

def get_classification_model(configs, device):
    model_name = configs.model.type

    if model_name == 'clip':
        return Clip(configs, device)
    
    if model_name == 'text':
        return BertTweetClassifier(configs, device)

    if model_name == 'combined':
        return ClipBertTweetClassifier(configs, device)

    raise Exception('Invalid model name')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='ckg11.json', required=True, help='Config file from src/configs/classifier')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    

    configs = read_json_configs(os.path.join('./src/configs/classifier', args.config))
    logger = FileLogger(configs)

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    train_dataset = MisogynyDataset(configs, './data/extracted/TRAINING', 'training.csv')
    test_dataset = MisogynyDataset(configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')
    
    model = get_classification_model(configs, args.device)
    trainer = MamiTrainer(configs, model, train_dataset, test_dataset, args.device, logger)
    trainer.train()
