import torch as t
from transformers import AutoModel, CLIPProcessor, CLIPModel
from PIL import Image


class ClipBertTweetClassifier(t.nn.Module):
    def __init__(self, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.linear_one = t.nn.Linear(1280, 512).to(device)
        self.linear_two = t.nn.Linear(512, 128).to(device)
        self.linear_three = t.nn.Linear(128, 5).to(device)

    def forward(self, input):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)
        images = [Image.open(image_path) for image_path in input['image']]

        with t.no_grad():
            _, pooled_output = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)

        combined_features = t.cat((pooled_output, features), 1)
        x_one = t.relu(self.linear_one(combined_features))
        x_two = t.relu(self.linear_two(x_one))
        return t.sigmoid(self.linear_three(x_two))
