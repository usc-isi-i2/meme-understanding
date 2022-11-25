from src.datasets.meme import MemeDataset

def get_train_dataset(configs):
    if configs.datasets.name == 'mami':
        return MemeDataset.create_mami_dataset_from_files('train', configs, './data/extracted/TRAINING', 'training.csv')
    
    if configs.datasets.name == 'hateful':
        return MemeDataset.create_hatefull_meme_dataset_from_files('train', configs, './data/extracted_hateful_meme/data', 'train.jsonl')

    raise Exception('Invalid dataset name')

def get_test_dataset(configs):
    if configs.datasets.name == 'mami':
        return MemeDataset.create_mami_dataset_from_files('test', configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')
    
    if configs.datasets.name == 'hateful':
        return MemeDataset.create_hatefull_meme_dataset_from_files('test', configs, './data/extracted_hateful_meme/data', 'dev.jsonl')

    raise Exception('Invalid dataset name')