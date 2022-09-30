from transformers import CLIPProcessor, CLIPModel
import torch as t

class Clip(t.nn.Module):
    def __init__(self, model_path="openai/clip-vit-base-patch32", device='cpu') -> None:
        super(Clip, self).__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_model = CLIPModel.from_pretrained(model_path).to(device)
        self.linear = t.nn.Linear(512, 1).to(device)

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        
        return t.sigmoid(self.linear(features))

