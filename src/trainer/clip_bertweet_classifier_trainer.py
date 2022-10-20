from PIL import Image
from tqdm import tqdm


from torch.nn import BCELoss
from torch.optim import Adam
from torch import Tensor

from src.trainer.trainer import Trainer
from src.models.clip_bertweet_classifier import ClipBertTweetClassifier

class ClipBertweetClassifierTrainer(Trainer):
    def __init__(self, train_dataset, test_dataset, device) -> None:
        model = ClipBertTweetClassifier(device)
        super().__init__(model, train_dataset, test_dataset, device)


    def train(self):
        bce_loss = BCELoss()

        self.model.train()
        for epoch in range(50):
            print('*' * 50)
            correct = 0
            total_loss = 0

            optimizer = Adam(self.model.parameters(), lr=0.0001)
            for batch in tqdm(self.train_dataloader):
                pred = self.model(batch['input'])

                predicted_label = [str(x[0]) for x in (pred > 0.5).int().tolist()]
                actual_label = batch['output']['misogynous']
                actual = Tensor([[float(a)] for a in actual_label]).to(self.device)            
                
                loss = bce_loss(pred, actual)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for a, p in zip(actual_label, predicted_label):
                    correct += 1 if a == p else 0

            print(f'Epoch {epoch}: Train Accuracy: {correct/self.train_dataset_length}')
            print(f'Epoch {epoch}: Train Loss: {total_loss}')
            self.eval(epoch)


    def eval(self, epoch):
        correct = 0
        self.model.eval()
        for batch in tqdm(self.test_dataloader):
            pred = self.model(batch['input'])

            predicted_label = [str(x[0]) for x in (pred > 0.5).int().tolist()]
            actual_label = batch['output']['misogynous']
    
            for a, p in zip(actual_label, predicted_label):
                correct += 1 if a == p else 0

        print(f'Epoch {epoch}: Test Accuracy: {correct/self.test_dataset_length}')
            

    def predict(self):
        return super().predict()