from PIL import Image
from tqdm import tqdm


from torch.nn import BCELoss
from torch.optim import Adam
from torch import Tensor

from src.datasets.mami import output_keys
from src.trainer.trainer import Trainer
from src.utils.mami import calculate

class MamiTrainer(Trainer):
    def __init__(self, model, train_dataset, test_dataset, device) -> None:
        super().__init__(model, train_dataset, test_dataset, device)


    def train(self):
        bce_loss = BCELoss()

        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(100):
            print('*' * 50)
            correct = {k:0 for k in output_keys}
            total_loss = 0

            for batch in tqdm(self.train_dataloader):
                pred = self.model(batch['input'])
                actual_output = calculate(pred, batch['output'], correct)
                actual_output = Tensor(actual_output).to(self.device)                            
                loss = bce_loss(pred, actual_output)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch}: Train Loss: {total_loss}')
            for output_key in output_keys:
                print(f'Epoch {epoch}: Train Accuracy: {output_key}: {correct[output_key]/self.train_dataset_length}')
                
            self.eval(epoch)


    def eval(self, epoch):
        self.model.eval()

        correct = {k:0 for k in output_keys}
        for batch in tqdm(self.test_dataloader):
            pred = self.model(batch['input'])
            calculate(pred, batch['output'], correct)

        for output_key in output_keys:
                print(f'Epoch {epoch}: Test Accuracy: {output_key}: {correct[output_key]/self.test_dataset_length}')
            

    def predict(self):
        return super().predict()