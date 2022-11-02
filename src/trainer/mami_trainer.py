import json
from turtle import pos
from tqdm import tqdm

from sklearn.metrics import classification_report

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch import Tensor, sigmoid

from src.datasets.mami import output_keys
from src.trainer.trainer import Trainer
from src.utils.mami import calculate

class MamiTrainer(Trainer):
    def __init__(self, configs, model, train_dataset, test_dataset, device, logger) -> None:
        super().__init__(configs, model, train_dataset, test_dataset, device, logger)

    def update_best(self, scores):
        sum_scores = 0
        for output_key in output_keys:
            sum_scores += scores[output_key][output_key]['f1-score']

        score = sum_scores / len(output_keys)

        if not (self.best_score) or (score > self.best_score):
            self.best_score = score

            print(f'-'*20)
            print(f'Best score: {score}')
            print(f'-'*20)

            return True
        
        return False

    def train(self):
        pos_weights = Tensor([0.5/(self.configs.datasets.mami.train.configs[k]) for k in output_keys]).to(self.device)
        bce_loss = BCEWithLogitsLoss(pos_weight=pos_weights)

        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=0.0001)
        
        for epoch in range(self.configs.train.epochs):
            print('*' * 50)
            actual_labels = {k:[] for k in output_keys}
            predicted_labels = {k:[] for k in output_keys}
            total_loss = 0

            for batch in tqdm(self.train_dataloader):
                pred = self.model(batch['input'])
                actual_output = calculate(pred, batch['output'], actual_labels, predicted_labels)
                actual_output = Tensor(actual_output).to(self.device)                            
                loss = bce_loss(pred, actual_output)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            log_dict = {'epoch': epoch, 'type': 'train'}
            for k in output_keys:
                log_dict[k] = classification_report(actual_labels[k], predicted_labels[k], target_names=[f'!{k}', k], output_dict=True)
            
            self.logger.log_file(log_dict)
            self.logger.log_file({k: log_dict[k][k]['f1-score'] for k in output_keys})
            self.logger.log_console({k: log_dict[k][k]['f1-score'] for k in output_keys})
    
            self.eval(epoch)


    def eval(self, epoch):
        self.model.eval()
        actual_labels = {k:[] for k in output_keys}
        predicted_labels = {k:[] for k in output_keys}
        
        predictions = {}
        for batch in tqdm(self.test_dataloader):
            pred = self.model(batch['input'])
            calculate(pred, batch['output'], actual_labels, predicted_labels)

            for image_path, scores in zip(batch['input']['image'] , sigmoid(pred).tolist()):
                predictions[image_path] = {k: v for k, v in zip(output_keys, scores)}

        log_dict = {'epoch': epoch, 'type': 'test'}
        for k in output_keys:
            log_dict[k] = classification_report(actual_labels[k], predicted_labels[k], target_names=[f'!{k}', k], output_dict=True)

        if self.update_best(log_dict):
            print(f'$$$ Last best prediction at epoch: {epoch}')
            with open(self.configs.predictions.filepath, 'w') as f:
                json.dump(predictions, f)

            with open(self.configs.logs.best_metrics, 'w') as f:
                json.dump(log_dict, f)
        
        self.logger.log_file(log_dict)
        self.logger.log_file({k: log_dict[k][k]['f1-score'] for k in output_keys})
        self.logger.log_console({k: log_dict[k][k]['f1-score'] for k in output_keys})
    
    def predict(self):
        pass