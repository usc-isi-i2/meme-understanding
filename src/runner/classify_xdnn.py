import os
from argparse import ArgumentParser
import json

from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from src.configs.config_reader import read_json_configs
from src.datasets.utils import get_train_dataset, get_test_dataset
from src.models.utils import get_classification_model
from src.models.xDNN.xDNN_class import xDNN as xDNN
from src.models.xDNN.xDNN_class_softmax import xDNN as xDNN_softmax
from src.logger import Logger
from src.trainer.mami_trainer import MamiTrainer

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='ckg11.json', required=True, help='Config file from src/configs/classifier')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./src/configs/xdnn', args.config))
    logger = Logger(configs)

    train_dataset = get_train_dataset(configs)
    test_dataset = get_test_dataset(configs)
    
    trainer = MamiTrainer(get_classification_model, configs, train_dataset, test_dataset, args.device, logger)

    train_features = trainer.extract_features(train_dataset)
    test_features = trainer.extract_features(test_dataset)

    run_xDNN_softmax = False

    if run_xDNN_softmax:
        scaler_train = MinMaxScaler()
        train_features_vector = np.array(train_features['Features'])
        scaler_train.fit(train_features_vector)
        train_features_vector = scaler_train.transform(train_features_vector)

        input_learning = {
            'Images': train_features_vector,
            'Features': np.array(train_features['Features']),
            'Labels': np.array(train_features['Labels'])
        }

        output_learning = xDNN_softmax(input_learning, 'Learning')

        scaler_test = MinMaxScaler()
        test_features_vector = np.array(test_features['Features'])
        scaler_test.fit(test_features_vector)
        test_features_vector = scaler_test.transform(test_features_vector)

        input_testing = {
            'xDNNParms': output_learning['xDNNParms'],
            'Images': test_features_vector,
            'Features': np.array(test_features['Features']),
            'Labels': np.array(test_features['Labels'])
        }

        output_validation = xDNN_softmax(input_testing, 'Validation')


    else:
        input_learning = {
            'Images': np.array(train_features['Images']),
            'Features': np.array(train_features['Features']),
            'Labels': np.array(train_features['Labels'])
        }

        output_learning = xDNN(input_learning,'Learning')

        input_testing = {
            'xDNNParms': output_learning['xDNNParms'],
            'Images': np.array(test_features['Images']),
            'Features': np.array(test_features['Features']),
            'Labels': np.array(test_features['Labels'])
        }

        output_validation = xDNN(input_testing, 'Validation')

    # Log classwise prototype counts
    logger.log_text(configs.logs.files.xdnn, f'Training Dataset Length: {len(train_dataset)}')
    for k in output_learning['xDNNParms']['Parameters']:
        p_count = len(output_learning['xDNNParms']['Parameters'][k]['Prototype'])
        logger.log_text(configs.logs.files.xdnn, f'Class {k} has {p_count} prototypes')
        
    # Compute and log classification report
    predicted_labels = [int(x) for x in output_validation['EstLabs']]
    actual_labels = test_features['Labels']
    test_memes = output_validation['image_list']
    p_memes = output_validation['image_prototypes']

    prediction_data = {}
    for p, a, t, p_m in zip(predicted_labels, actual_labels, test_memes, p_memes):
        prediction_data[t] = {
            'predicted': p,
            'actual': a,
            'prototype': p_m
        }

    classification_report = classification_report(test_features['Labels'], predicted_labels, output_dict=True)
    logger.log_text(configs.logs.files.xdnn, str(classification_report))
    logger.log_text(configs.logs.files.xdnn, json.dumps(prediction_data))


    print(classification_report['macro avg']['f1-score'])

    print('done')
