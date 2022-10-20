from abc import abstractmethod, ABC

from torch.utils.data import DataLoader

class Trainer(ABC):
    def __init__(self, model, train_dataset, test_dataset, device, train_batch_size=4, test_batch_size=4) -> None:
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        self.device = device
        self.train_dataset_length = len(train_dataset)
        self.test_dataset_length = len(test_dataset)
        
    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def eval(self, dataset):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass