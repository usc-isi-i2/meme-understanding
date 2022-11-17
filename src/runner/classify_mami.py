import os
from argparse import ArgumentParser

import torch
import random

from src.configs.config_reader import read_json_configs
from src.datasets.meme import MemeDataset
from src.logger import Logger
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

def get_train_dataset(configs):
    if configs.datasets.name == 'mami':
        return MemeDataset.create_mami_dataset_from_files('train', configs, './data/extracted/TRAINING', 'training.csv')
    
    if configs.datasets.name == 'hateful':
        return MemeDataset.create_hatefull_meme_dataset_from_files('train', configs, './data/extracted_hateful_meme/data', 'train.jsonl')

    raise Exception('Invalid dataset name')

def get_test_dataset(configs):
    if configs.datasets.name == 'mami':
        return MemeDataset.create_mami_dataset_from_files('test', configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')
    
    if configs.datasets.name == 'hateful':
        return MemeDataset.create_hatefull_meme_dataset_from_files('test', configs, './data/extracted_hateful_meme/data', 'dev.jsonl')

    raise Exception('Invalid dataset name')


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
