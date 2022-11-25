import os
from argparse import ArgumentParser

from src.configs.config_reader import read_json_configs
from src.datasets.utils import get_train_dataset, get_test_dataset
from src.models.utils import get_classification_model
from src.logger import Logger
from src.trainer.mami_trainer import MamiTrainer

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='ckg11.json', required=True, help='Config file from src/configs/classifier')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./src/configs/classifier', args.config))
    logger = Logger(configs)

    train_dataset = get_train_dataset(configs)
    test_dataset = get_test_dataset(configs)

    model = get_classification_model(configs, args.device)
    
    trainer = MamiTrainer(get_classification_model, configs, train_dataset, test_dataset, args.device, logger)

    train_features = trainer.extract_features(train_dataset)
    test_features = trainer.extract_features(test_dataset)

    print('Train features shape: ', len(train_features['Features'][0]))
    print('Test features shape: ', len(test_features['Features'][0]))

    print('done')







