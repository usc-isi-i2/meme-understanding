from cgitb import text
from turtle import forward
import torch as t
from transformers import AutoModel


class BertTweetClassifier(t.nn.Module):
    def __init__(self, device='cpu') -> None:
        super().__init__()

        self.max_len = 128
        self.device = device
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
        self.linear_one = t.nn.Linear(768, 64).to(device)
        self.linear_two = t.nn.Linear(64, 1).to(device)

    def forward(self, input):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)

        with t.no_grad():
            _, pooled_output = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        x = t.relu(self.linear_one(pooled_output))
        return t.sigmoid(self.linear_two(x))
