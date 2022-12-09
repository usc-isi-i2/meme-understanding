from collections import defaultdict
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

class ClipKNN:
    def __init__(self, configs, model, device, train_dataset, test_dataset, targets) -> None:
        self.model = model
        self.configs = configs
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.targets = targets

        self.sorted_similarities = defaultdict(lambda: {})
        self.train_features = {}
        self.test_features = {}
    
    def extract_train_features(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(train_dataloader):
                features = self.model.get_intermediate_features(batch['input'], self.configs.model.knn.feature_layer)
                file_path = batch['input']['image']
                for i, f in enumerate(file_path):
                    self.train_features[f] = {'features': features[i], 'labels': {k: batch['output'][k][i] for k in batch['output'].keys()}}

    def extract_test_features(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                features = self.model.get_intermediate_features(batch['input'], self.configs.model.knn.feature_layer)
                file_path = batch['input']['image']
                for i, f in enumerate(file_path):
                    self.test_features[f] = {'features': features[i], 'labels': {k: batch['output'][k][i] for k in batch['output'].keys()}}

    def compute_similarities(self, save_to_file=True):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        similarities = defaultdict(lambda: {})
        for train_image_path, train_image_data in tqdm(self.train_features.items()):
            for test_image_path, test_image_data in self.test_features.items():
                similarities[test_image_path][train_image_path] = cos(train_image_data['features'], test_image_data['features']).item()

        for key, value in tqdm(similarities.items()):
            self.sorted_similarities[key] = dict(sorted(value.items(), key=lambda item: item[1], reverse=True)[:100])

        if save_to_file:
            with open(f'./data/processed/{self.configs.title}_sorted_similarities.json', 'w') as f:
                json.dump(self.sorted_similarities, f)

    def knn_classification(self, k=10, threshold=0.5, output_dict=False):
        y_true = []
        y_predicted = []

        for sample in self.test_dataset:
            image = sample['input']['image']

            votes = 0
            for s in list(self.sorted_similarities[image].keys())[:k]:
                votes += int(self.train_features[s]['labels']['misogynous'])

                prediction = 1 if votes/k >= threshold else 0
                y_predicted.append(str(prediction))
                y_true.append(sample['output']['misogynous'])

        
        report = classification_report(y_true, y_predicted, target_names=self.targets, output_dict=output_dict)
        if not output_dict:
            print(f'Running with k={k} and threshold={threshold}')
            print('\n')
            print(report)
        
        return report
