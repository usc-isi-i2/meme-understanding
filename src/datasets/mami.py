import os
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

output_keys = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']

class MisogynyDataset(Dataset):
  def __init__(self, data_dir, text_file, labels_file_path=None) -> None:
    self.data_dir = data_dir
    self.data_dict = defaultdict(lambda: {})
    self.update_data_from_file(os.path.join(data_dir, text_file))

    if labels_file_path:
      self.update_data_from_file(labels_file_path)

    self.data = list(self.data_dict.values())

    self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]

    encoding = self.tokenizer.encode_plus(
      item['Text Transcription'],
      add_special_tokens=True,
      max_length=128,
      truncation= True,
      return_token_type_ids=False,
      padding = 'max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    input = {
        'image': os.path.join(self.data_dir, item['file_name']),
        'text': item['Text Transcription'],
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    }

    output = {k:v for k,v in item.items() if k in output_keys}
    return {'input': input, 'output': output}
    

  def update_data_from_file(self, filepath):
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
      keys = None if '.csv' in filepath else ['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
      for line in f:
        if not keys:
          keys = line.strip().split('\t')
          continue
        
        filename = None
        for index, value in enumerate(line.strip().split('\t')):
          if index == 0:
            filename = value

          self.data_dict[filename][keys[index]] = value