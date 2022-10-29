import json
from tqdm import tqdm

from sklearn.metrics import classification_report

from torch.nn import BCELoss
from torch.optim import Adam
from torch import Tensor

from src.datasets.mami import output_keys
from src.trainer.trainer import Trainer
from src.utils.mami import calculate

class MamiTrainer(Trainer):
    def __init__(self, configs, model, train_dataset, test_dataset, device, logger) -> None:
        super().__init__(configs, model, train_dataset, test_dataset, device, logger)


    def train(self):
        bce_loss = BCELoss()

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
                log_dict[k] = classification_report(actual_labels[k], predicted_labels[k], target_names=[k, f'!{k}'], output_dict=True)
            
            self.logger.log(log_dict)
    
            self.eval(epoch)


    def eval(self, epoch):
        self.model.eval()
        actual_labels = {k:[] for k in output_keys}
        predicted_labels = {k:[] for k in output_keys}
        
        predictions = {}
        for batch in tqdm(self.test_dataloader):
            pred = self.model(batch['input'])
            calculate(pred, batch['output'], actual_labels, predicted_labels)

            for image_path, scores in zip(batch['input']['image'] , pred.tolist()):
                predictions[image_path] = {k: v for k, v in zip(output_keys, scores)}

        log_dict = {'epoch': epoch, 'type': 'test'}
        for k in output_keys:
            log_dict[k] = classification_report(actual_labels[k], predicted_labels[k], target_names= [k, f'!{k}'], output_dict=True)

        if not (self.best_score) or (self.best_score < log_dict[output_keys[0]]['weighted avg']['f1-score']):
            self.best_score = log_dict[output_keys[0]]['weighted avg']['f1-score']
            print(f'$$$ Last best prediction at epoch: {epoch}')
            with open(self.configs.predictions.filepath, 'w') as f:
                json.dump(predictions, f)

            with open(self.configs.logs.best_metrics, 'w') as f:
                json.dump(log_dict, f)
        
        self.logger.log(log_dict)

    def predict(self):
        return super().predict()