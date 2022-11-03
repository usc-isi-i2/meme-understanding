from argparse import ArgumentParser
import json
import os
import shutil

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import torch
from tqdm import tqdm

from src.datasets.mami import MisogynyDataset
from src.configs.config_reader import read_json_configs

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--configs', required=True, help='configs file from src/configs directory')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    args = arg_parser.parse_args()

    configs = read_json_configs(args.configs)

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    
    model.to(args.device)

    train_dataset = MisogynyDataset(configs, './data/extracted/TRAINING', 'training.csv')
    test_dataset = MisogynyDataset(configs, './data/extracted/test', 'Test.csv', './data/extracted/test_labels.txt')

    detected_object_directory = configs.detected_object_directory

    if os.path.exists(detected_object_directory):
        shutil.rmtree(detected_object_directory)

    os.makedirs(detected_object_directory)

    detected_objects_info = {}
    failed_images = []
    detected_object_index = 0
    for sample in tqdm(train_dataset + test_dataset):
        try:
            image = Image.open(sample['input']['image'])
            inputs = feature_extractor(images=image, return_tensors="pt").to(args.device)
            outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(args.device)
            results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

            objects_for_current_sample = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                # let's only keep detections with score > 0.9
                if score > 0.9:
                    detected_object_index += 1
                    ni = image.crop(box)
                    detected_object_filepath = f'{detected_object_directory}/{detected_object_index}.jpg'
                    ni.save(detected_object_filepath)
                    objects_for_current_sample.append({
                        'filename': detected_object_filepath,
                        'confidence': round(score.item(), 3),
                        'label': model.config.id2label[label.item()]
                    })
            
            detected_objects_info[sample['input']['image']] = objects_for_current_sample
        
        except Exception as e:
            failed_images.append({sample['input']['image']: str(e)})
    
    detected_objects_info['failed'] = failed_images
    info_filepath = os.path.join(configs.processed_data_path, configs.objects_info_filename)
    with open(info_filepath, 'w') as f:
        print(f'Saving file information at: {info_filepath}')
        json.dump(detected_objects_info, f)
