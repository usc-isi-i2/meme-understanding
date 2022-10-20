from os import environ
from argparse import ArgumentParser

from src.datasets.mami import MisogynyDataset
from src.trainer.bertweet_classifier_trainer import BertweetClassifierTrainer

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    args = arg_parser.parse_args()

    train_dataset = MisogynyDataset('./data/extracted/TRAINING', 'training.csv')
    test_dataset = MisogynyDataset('./data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

    trainer = BertweetClassifierTrainer(test_dataset, test_dataset, args.device)
    trainer.train()

    