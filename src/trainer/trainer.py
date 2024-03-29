from abc import abstractmethod, ABC
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.logger import Logger

class Trainer(ABC):
    def __init__(self, get_model_func, configs, train_dataset, test_dataset, device, logger) -> None:
        self.get_model_func = get_model_func
        self.configs = configs
        self.model = None
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.logger: Logger = logger
        self.model_save_dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models)

    def train_kfold(self):
        self.logger.log_text(self.configs.logs.files.data, self.train_dataset.summarize())
        self.logger.log_text(self.configs.logs.files.data, self.test_dataset.summarize())


        test_dataloader = DataLoader(self.test_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
        kfold_test_metrics = []
        for kth_fold in range(self.configs.train.k_fold):
            self.model = self.get_model_func(self.configs, self.device)
            self.optimizer = Adam(self.model.parameters(), lr=0.0001)

            train_set, eval_set = self.train_dataset.get_kth_fold_dataset(kth_fold)
            self.logger.log_text(self.configs.logs.files.data, train_set.summarize())
            self.logger.log_text(self.configs.logs.files.data, eval_set.summarize())

            train_dataloader = DataLoader(train_set, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            best_score = None
            best_parames = {}
            epcohs_without_improvement = 0

            test_predictions = None
            test_metric = None
            for epoch in range(self.configs.train.epochs):
                self.train(train_dataloader)
                train_scores, _ = self.eval(train_dataloader)
                eval_scores, _ = self.eval(eval_dataloader)

                eval_metric = self.summarize_scores(eval_scores, eval_set.get_class_distribution(True))
                if best_score is None or  eval_metric > best_score:
                    best_score = eval_metric
                    test_scores, test_predictions = self.eval(test_dataloader)
                    test_metric = self.summarize_scores(test_scores, self.test_dataset.get_class_distribution(True))
                    best_parames = {
                        'kth_fold': kth_fold,
                        'epoch': epoch,
                        'eval_metric': eval_metric,
                        'test_metric': test_metric,
                        'test': {k: test_scores[k]['macro avg']['f1-score'] for k in self.configs.datasets.labels},
                        'eval': {k: eval_scores[k]['macro avg']['f1-score'] for k in self.configs.datasets.labels},
                        'eval_scores': eval_scores,
                        'test_scores': test_scores,
                    }

                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f'best_model_{kth_fold}.pt'))
                    self.logger.log_file(self.configs.logs.files.best, best_parames)
                    epcohs_without_improvement = 0
                else:
                    epcohs_without_improvement += 1
                
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'train': {k: train_scores[k][k]['f1-score'] for k in self.configs.datasets.labels}})
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'eval': {k: eval_scores[k][k]['f1-score'] for k in self.configs.datasets.labels}})

                if epcohs_without_improvement >= self.configs.train.patience:
                    break
            
            kfold_test_metrics.append(test_metric)
            self.logger.log_file(self.configs.logs.files.predictions, test_predictions)
        
        self.logger.log_file(self.configs.logs.files.best, {'avg': sum(kfold_test_metrics)/len(kfold_test_metrics), 'kfold_test_metrics': kfold_test_metrics})
        

    @abstractmethod
    def summarize_scores(self, scores):
        pass
        
    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def eval(self, dataset):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass