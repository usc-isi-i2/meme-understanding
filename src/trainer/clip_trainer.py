from PIL import Image
from tqdm import tqdm


from torch.nn import BCELoss
from torch.optim import Adam
from torch import Tensor
import torch

from src.trainer.trainer import Trainer
from src.models.clip import Clip

class ClipTrainer(Trainer):
    def __init__(self, model_path, train_dataset, test_dataset, device) -> None:
        model = Clip(model_path, device)
        super().__init__(model, train_dataset, test_dataset, device)


    def train(self):
        bce_loss = BCELoss()

        self.model.train()
        for epoch in range(50):
            print('*' * 50)
            correct = 0
            total_loss = 0

            optimizer = Adam(self.model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
            for batch in tqdm(self.train_dataloader):
                image = [Image.open(image_path) for image_path in batch['input']['image']][0]
                pred = self.model(image)

                predicted_label = int((pred > 0.5).int().tolist()[0][0])
                actual_label = int(batch['output']['misogynous'][0])
                actual = Tensor([[actual_label]]).to(self.device)            
                
                loss = bce_loss(pred, actual)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += 1 if actual_label == predicted_label else 0

            print(f'Epoch {epoch}: Train Accuracy: {correct/len(self.train_dataloader)}')
            print(f'Epoch {epoch}: Train Loss: {total_loss}')
            self.eval(epoch)



    def eval(self, epoch):
        correct = 0
        self.model.eval()
        for batch in tqdm(self.test_dataloader):
            image = [Image.open(image_path) for image_path in batch['input']['image']][0]
            pred = self.model(image)

            predicted_label = int((pred > 0.5).int().tolist()[0][0])
            actual_label = int(batch['output']['misogynous'][0])
    
            correct += 1 if actual_label == predicted_label else 0

        print(f'Epoch {epoch}: Test Accuracy: {correct/len(self.test_dataloader)}')
            

    def predict(self):
        return super().predict()