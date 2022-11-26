import os
from argparse import ArgumentParser

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.config_reader import read_json_configs
from src.datasets.mnist import MNISTDataset
from src.models.utils import get_classification_model
from src.models.xDNN.xDNN_class import xDNN as xDNN
from src.models.xDNN.xDNN_class_softmax import xDNN as xDNN_softmax
from src.logger import Logger

def get_features(model, dataset, limit):
    model.eval()

    dataloader = DataLoader(dataset, 258, shuffle=True)

    features = {}

    outputs = []
    features['Images'] = []
    features['Labels'] = []

    count = 0
    for batch in tqdm(dataloader):
        images = [dataset.get_image(index) for index in batch['index']]
        f = model.get_pil_image_features(images)
        features['Images'].extend([[f'named_index_{int(x)}'] for x in batch['index']])
        outputs.extend(f)
        features['Labels'].extend([int(x) for x in batch['Label']])
        count += 258
        if count > limit:
            break

    result = torch.stack(outputs)
    normalized = torch.nn.functional.normalize(result, p=2, dim=1)

    features['Features'] = normalized.cpu().numpy()
    return features
    

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='ckg11.json', required=True, help='Config file from src/configs/classifier')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./src/configs/classifier', args.config))
    logger = Logger(configs)

    train_dataset = MNISTDataset('./data/mnist/train', train=True)
    test_dataset =  MNISTDataset('./data/mnist/test', train=False)

    clip_model = get_classification_model(configs, args.device)
    
    print('Training dataset length: ', len(train_dataset))
    print('Test dataset length: ', len(test_dataset))
    

    train_features = get_features(clip_model, train_dataset, 10000)
    test_features = get_features(clip_model, test_dataset, 1000)

    run_xDNN_softmax = True

    if run_xDNN_softmax:
        scaler_train = MinMaxScaler()
        train_features_vector = train_features['Features']
        scaler_train.fit(train_features_vector)
        train_features_vector = scaler_train.transform(train_features_vector)

        input_learning = {
            'Images': train_features['Images'],
            'Features': train_features_vector,
            'Labels': np.array(train_features['Labels'])
        }

        output_learning = xDNN_softmax(input_learning, 'Learning')

        scaler_test = MinMaxScaler()
        test_features_vector = test_features['Features']
        scaler_test.fit(test_features_vector)
        test_features_vector = scaler_test.transform(test_features_vector)

        input_testing = {
            'xDNNParms': output_learning['xDNNParms'],
            'Images': test_features['Images'],
            'Features': test_features_vector,
            'Labels': np.array(test_features['Labels'])
        }

        output_testing = xDNN_softmax(input_testing, 'Validation')


    else:
        input_learning = {
            'Images': np.array(train_features['Images']),
            'Features': train_features['Features'],
            'Labels': np.array(train_features['Labels'])
        }

        output_learning = xDNN(input_learning,'Learning')

        input_testing = {
            'xDNNParms': output_learning['xDNNParms'],
            'Images': np.array(test_features['Images']),
            'Features': test_features['Features'],
            'Labels': np.array(test_features['Labels'])
        }

        output_validation = xDNN(input_testing, 'Validation')

    print('done')
