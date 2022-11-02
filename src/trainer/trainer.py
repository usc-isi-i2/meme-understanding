from abc import abstractmethod, ABC

from torch.utils.data import DataLoader

from src.logger import Logger

class Trainer(ABC):
    def __init__(self, configs, model, train_dataset, test_dataset, device, logger, train_batch_size=3, test_batch_size=32) -> None:
        self.configs = configs
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=configs.train.train_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=configs.train.eval_batch_size)
        self.device = device
        self.train_dataset_length = len(train_dataset)
        self.test_dataset_length = len(test_dataset)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.logger: Logger = logger

        self.best_score = None

    @abstractmethod
    def update_best(self, scores) -> bool:
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