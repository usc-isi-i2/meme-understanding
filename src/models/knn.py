from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report

class ClipKNN:
    def __init__(self, model, processor, device, train_dataset, test_dataset, targets) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.targets = targets

        self.sorted_similarities = defaultdict(lambda: {})
        self.train_features = {}
        self.test_features = {}

        self.model.to(device)
    
    def extract_train_features(self):
        with torch.no_grad():
            for sample in tqdm(self.train_dataset):
                file_path = sample['input']['image']
                image = Image.open(sample['input']['image'])
                labels = sample['output']
                inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_image_features(**inputs)

                self.train_features[file_path] = {'labels': labels, 'features': features}

    def extract_test_features(self):
        with torch.no_grad():
            for sample in tqdm(self.test_dataset):
                file_path = sample['input']['image']
                image = Image.open(sample['input']['image'])
                labels = sample['output']
                inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_image_features(**inputs)

                self.test_features[file_path] = {'labels': labels, 'features': features}

    def compute_similarities(self):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = defaultdict(lambda: {})
        for train_image_path, train_image_data in tqdm(self.train_features.items()):
            for test_image_path, test_image_data in self.test_features.items():
                similarities[test_image_path][train_image_path] = cos(train_image_data['features'], test_image_data['features']).item()

        for key, value in tqdm(similarities.items()):
            self.sorted_similarities[key] = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))

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



    
