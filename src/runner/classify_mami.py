from ast import arg
from os import environ
from argparse import ArgumentParser

from src.datasets.mami import MisogynyDataset
from src.trainer.mami_trainer import MamiTrainer
from src.models.bertweet_classifier import BertTweetClassifier
from src.models.clip import Clip
from src.models.clip_bertweet_classifier import ClipBertTweetClassifier

def get_classification_model(model_name, device):
    if model_name == 'clip':
        return Clip(device)
    
    if model_name == 'bert':
        return BertTweetClassifier(device)

    if model_name == 'combined':
        return ClipBertTweetClassifier(device)

    raise Exception('Invalid model name')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model', default='bert', required=True, help='Supported models: clip/bert/combined')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    args = arg_parser.parse_args()

    train_dataset = MisogynyDataset('./data/extracted/TRAINING', 'training.csv')
    test_dataset = MisogynyDataset('./data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')
    
    model = get_classification_model(args.model, args.device)
    trainer = MamiTrainer(model, train_dataset, test_dataset, args.device)
    trainer.train()
