import os
import json
import shutil

class Logger:
    def __init__(self, configs):
        self.configs = configs
        self.dir = os.path.join(configs.logs.dir, configs.title)

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        
        os.makedirs(self.dir, exist_ok=True)
        json.dump(configs.configs, open(os.path.join(self.dir, 'configs.json'), 'w'))
        

    def log_file(self, log_file, log_dict):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'a+') as f:
            f.write('\n')
            f.write(json.dumps(log_dict))

    def log_console(self, log_dict):
        print(log_dict)
