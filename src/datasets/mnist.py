# Create mnist dataset
#
# Compare this snippet from src/runner/mnist_xdnn_classifier.py:
#

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.data = MNIST(data_dir, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return {
            'index': index,
            'Label': label
        }
    
    def get_image(self, index):
        return self.data[index][0]
