from transformers import CLIPProcessor, CLIPModel
import torch as t

class Clip(t.nn.Module):
    def __init__(self, model_path="openai/clip-vit-base-patch32", device='cpu') -> None:
        super(Clip, self).__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_model = CLIPModel.from_pretrained(model_path).to(device)
        self.linear_one = t.nn.Linear(512, 64).to(device)
        self.linear_two = t.nn.Linear(64, 1).to(device)

    def forward(self, image, train_clip=False):
        if train_clip:
          inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
          features = self.clip_model.get_image_features(**inputs)
        else:
          with t.no_grad():
            inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)
        
        x = t.relu(self.linear_one(features))
        return t.sigmoid(self.linear_two(x))

