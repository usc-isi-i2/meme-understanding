import os
import json

class Logger:
    def __init__(self, configs):
        self.filepath = os.path.join(configs.logs.filepath)
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

        with open(self.filepath, 'w') as f:
            f.write(f'configs: {configs}')

    def log_file(self, log_dict):
        with open(self.filepath, 'a+') as f:
            f.write('\n')
            f.write(json.dumps(log_dict))

    def log_console(self, log_dict):
        print(log_dict)
