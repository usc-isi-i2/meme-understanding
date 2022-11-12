from abc import abstractmethod, ABC

from torch.utils.data import DataLoader
from torch.optim import Adam
from src.logger import Logger
from src.datasets.mami import output_keys

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

    def train_kfold(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
        for kth_fold in range(self.configs.train.k_fold):
            self.model = self.get_model_func(self.configs, self.device)
            self.optimizer = Adam(self.model.parameters(), lr=0.0001)

            train_set, eval_set = self.train_dataset.get_kth_fold_dataset(kth_fold)
            train_dataloader = DataLoader(train_set, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            
            print('*' * 50)
            train_set.summarize()
            print('*' * 25)
            eval_set.summarize()
            print('*' * 50)

            best_score = None
            best_parames = {}
            epcohs_without_improvement = 0

            test_predictions = None
            for epoch in range(self.configs.train.epochs):
                self.train(train_dataloader)
                train_scores, _ = self.eval(train_dataloader)
                eval_scores, _ = self.eval(eval_dataloader)

                if best_score is None or self.summarize_scores(eval_scores) > best_score:
                    best_score = self.summarize_scores(eval_scores)
                    test_scores, test_predictions = self.eval(test_dataloader)
                    best_parames = {
                        'kth_fold': kth_fold,
                        'epoch': epoch,
                        'test': {k: test_scores[k][k]['f1-score'] for k in output_keys},
                        'eval': {k: eval_scores[k][k]['f1-score'] for k in output_keys},
                    }

                    self.logger.log_file(self.configs.logs.files.best, best_parames)
                    epcohs_without_improvement = 0
                else:
                    epcohs_without_improvement += 1
                
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'train': {k: train_scores[k][k]['f1-score'] for k in output_keys}})
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'eval': {k: eval_scores[k][k]['f1-score'] for k in output_keys}})

                if epcohs_without_improvement >= self.configs.train.patience:
                    break
            
            self.logger.log_file(self.configs.logs.files.predictions, test_predictions)
        

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