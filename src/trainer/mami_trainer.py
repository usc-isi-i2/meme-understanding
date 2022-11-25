import json
from turtle import pos
from tqdm import tqdm

from sklearn.metrics import classification_report

from torch.nn import BCEWithLogitsLoss
from torch import Tensor, sigmoid
from torch.utils.data import DataLoader

from src.trainer.trainer import Trainer
from src.utils.mami import calculate

class MamiTrainer(Trainer):
    def __init__(self, get_model_func, configs, train_dataset, test_dataset, device, logger) -> None:
        super().__init__(get_model_func, configs, train_dataset, test_dataset, device, logger)
        pos_weights = Tensor([0.5/(self.configs.datasets.train.configs[k]) for k in configs.datasets.labels]).to(self.device)
        self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weights)
        self.task = self.configs.task
        

    def summarize_scores(self, scores, label_distribution):
        if self.task == 'A':
            print('Computing scores for task A')
            return scores[self.configs.datasets.labels[0]]['macro avg']['f1-score']
        
        print('Computing scores for task B')
        sum_scores = 0
        sum_labels = 0
        for output_key in self.configs.datasets.labels[1:]:
            sum_scores += label_distribution[output_key]*scores[output_key]['macro avg']['f1-score']
            sum_labels += label_distribution[output_key]

        summarized_scores = sum_scores / sum_labels
        return summarized_scores

    def train(self, train_dataloader):
        self.model.train()
        print('*' * 50)
        actual_labels = {k:[] for k in self.configs.datasets.labels}
        predicted_labels = {k:[] for k in self.configs.datasets.labels}
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            pred = self.model(batch['input'])
            actual_output = calculate(pred, batch['output'], actual_labels, predicted_labels, self.configs.datasets.labels)
            actual_output = Tensor(actual_output).to(self.device)                            
            loss = self.bce_loss(pred, actual_output)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss

    def extract_features(self, dataset):
        model = self.get_model_func(self.configs, self.device)
        model.eval()

        dataloader = DataLoader(dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)

        features = {}
        features['Images'] = []
        features['Features'] = []
        features['Labels'] = []

        for batch in tqdm(dataloader):
            f = model.get_features(batch['input'])
            features['Images'].extend(batch['input']['image'])
            features['Features'].extend(f.tolist())
            features['Labels'].extend(batch['output'][self.configs.datasets.labels[0]])

        return features
        

    def eval(self, test_dataloader):
        self.model.eval()
        actual_labels = {k:[] for k in self.configs.datasets.labels}
        predicted_labels = {k:[] for k in self.configs.datasets.labels}

        predictions = {}
        for batch in tqdm(test_dataloader):
            pred = self.model(batch['input'])
            calculate(pred, batch['output'], actual_labels, predicted_labels, self.configs.datasets.labels)

            for image_path, scores in zip(batch['input']['image'] , sigmoid(pred).tolist()):
                predictions[image_path] = {k: v for k, v in zip(self.configs.datasets.labels, scores)}

        log_dict = {}
        for k in self.configs.datasets.labels:
            log_dict[k] = classification_report(actual_labels[k], predicted_labels[k], target_names=[f'!{k}', k], output_dict=True)

        return log_dict, predictions
    
    def predict(self, test_dataloader):
        self.model.eval()

        predictions = {}
        for batch in tqdm(test_dataloader):
            pred = self.model(batch['input'])

            for image_path, scores in zip(batch['input']['image'] , sigmoid(pred).tolist()):
                predictions[image_path] = {k: v for k, v in zip(self.configs.datasets.labels, scores)}

        return predictions