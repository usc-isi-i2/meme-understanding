import os
from collections import defaultdict
from torch.utils.data import Dataset

output_keys = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']

class MisogynyDataset(Dataset):
  def __init__(self, data_dir, text_file, labels_file_path=None) -> None:
    self.data_dir = data_dir
    self.data_dict = defaultdict(lambda: {})
    self.update_data_from_file(os.path.join(data_dir, text_file))

    if labels_file_path:
      self.update_data_from_file(labels_file_path)

    self.data = list(self.data_dict.values())

  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    input = {
        'image': os.path.join(self.data_dir, item['file_name']),
        'text': item['Text Transcription']
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