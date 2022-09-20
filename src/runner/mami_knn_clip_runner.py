from os import environ
from argparse import ArgumentParser

from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from tabulate import tabulate

from src.models.knn import ClipKNN
from src.datasets.mami import MisogynyDataset

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    args = arg_parser.parse_args()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = MisogynyDataset('./data/extracted/TRAINING', 'training.csv')
    test_dataset = MisogynyDataset('./data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

    target_names = ["Not misogyny", "Misogyny"]
    clip_knn_model = ClipKNN(model, processor, args.device, train_dataset, test_dataset, target_names)

    clip_knn_model.extract_train_features()
    clip_knn_model.extract_test_features()
    clip_knn_model.compute_similarities()

    headers = ['k', 'threshold', 'Presicion', 'Recall', 'F1']
    values = []
    for k in tqdm(range(1, 100)):
        for threshold in range(40, 99):
            report = clip_knn_model.knn_classification(k, threshold/100, True)['weighted avg']
            values.append([k, threshold, report['precision'], report['recall'], report['f1-score']])

    values = sorted(values, key=lambda metrics: metrics[-1], reverse=True)
    print(tabulate(values[:20], headers=headers))
