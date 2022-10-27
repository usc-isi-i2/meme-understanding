import os
import json

class FileLogger:
    def __init__(self, configs):
        self.filepath = os.path.join(configs.logs.filepath)
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

        with open(self.filepath, 'w') as f:
            f.write(f'configs: {configs}')

    def log(self, log_dict):
        with open(self.filepath, 'a+') as f:
            f.write('\n')
            f.write(json.dumps(log_dict))
