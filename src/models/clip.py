from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch as t

class Clip(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super(Clip, self).__init__()

        self.configs = configs
        self.device = device
        model_path="openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_model = CLIPModel.from_pretrained(model_path).to(device)
        self.linear_one = t.nn.Linear(512, 512).to(device)
        self.linear_two = t.nn.Linear(512, 256).to(device)
        self.linear_three = t.nn.Linear(256, 128).to(device)
        self.linear_four = t.nn.Linear(128, len(self.configs.datasets.labels)).to(device)

    def forward(self, input):
        images = [Image.open(image_path) for image_path in input['image']]

        with t.no_grad():
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)
        
        x = t.relu(self.linear_one(features))
        x = t.relu(self.linear_two(x))
        x = t.relu(self.linear_three(x))
        return self.linear_four(x)

    def get_features(self, input):
        images = [Image.open(image_path) for image_path in input['image']]

        with t.no_grad():
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)
        
        return features