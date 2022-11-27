import torch as t
from transformers import AutoModel, CLIPProcessor, CLIPModel
from PIL import Image


class ClipBertTweetClassifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.configs = configs
        self.device = device
        self.bert = AutoModel.from_pretrained(configs.model.text.bert).to(device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.linear_one = t.nn.Linear(1280, 512).to(device)
        self.linear_two = t.nn.Linear(512, 256).to(device)
        self.linear_three = t.nn.Linear(256, 128).to(device)
        self.linear_four = t.nn.Linear(128, len(self.configs.datasets.labels)).to(device)

    def forward(self, input):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)
        images = [Image.open(image_path) for image_path in input['image']]

        with t.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)

        combined_features = t.cat((pooled_output, features), 1)
        x = t.relu(self.linear_one(combined_features))
        x = t.relu(self.linear_two(x))
        x = t.relu(self.linear_three(x))
        return self.linear_four(x)
    
    def get_intermediate_features(self, input, layer):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)
        images = [Image.open(image_path) for image_path in input['image']]

        with t.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.clip_model.get_image_features(**inputs)

        combined_features = t.cat((pooled_output, features), 1)
        x = t.relu(self.linear_one(combined_features))
        if layer == 1:
            return x
        x = t.relu(self.linear_two(x))
        if layer == 2:
            return x
        x = t.relu(self.linear_three(x))
        if layer == 3:
            return x

        return self.linear_four(x)
