from src.models.clip import Clip
from src.models.bertweet_classifier import BertTweetClassifier
from src.models.clip_bertweet_classifier import ClipBertTweetClassifier

def get_classification_model(configs, device):
    model_name = configs.model.type

    if model_name == 'clip':
        return Clip(configs, device)
    
    if model_name == 'text':
        return BertTweetClassifier(configs, device)

    if model_name == 'combined':
        return ClipBertTweetClassifier(configs, device)

    raise Exception('Invalid model name')
