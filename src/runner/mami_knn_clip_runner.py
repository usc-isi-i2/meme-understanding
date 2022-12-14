import os
from os import environ
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from tabulate import tabulate

from src.models.knn import ClipKNN
from src.models.utils import get_classification_model
from src.datasets.meme import MemeDataset
from src.configs.config_reader import read_json_configs

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    arg_parser.add_argument('--configs', required=True, help='configs file from src/configs directory')
    args = arg_parser.parse_args()

    configs = read_json_configs(args.configs)
    
    model_save_dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models)
    model = get_classification_model(configs, args.device)
    torch.load(os.path.join(model_save_dir, f'best_model_{configs.model.knn.best_fold}.pt'))
    
    train_dataset = MemeDataset.create_mami_dataset_from_files('train' , configs, 'data/extracted/TRAINING', 'training.csv')
    test_dataset = MemeDataset.create_mami_dataset_from_files('test', configs, 'data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

    target_names = ["Not misogyny", "Misogyny"]
    clip_knn_model = ClipKNN(configs, model, args.device, train_dataset, test_dataset, target_names)

    clip_knn_model.extract_train_features()
    clip_knn_model.extract_test_features()
    clip_knn_model.compute_similarities()

    headers = ['k', 'threshold', 'Presicion', 'Recall', 'F1']
    values = []
    for k in tqdm(range(1, 50)):
        for threshold in range(40, 99, 5):
            report = clip_knn_model.knn_classification(k, threshold/100, True)['macro avg']
            values.append([k, threshold, report['precision'], report['recall'], report['f1-score']])

    values = sorted(values, key=lambda metrics: metrics[-1], reverse=True)
    print(tabulate(values[:20], headers=headers))
