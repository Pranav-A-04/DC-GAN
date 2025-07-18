import glob
import os
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, split, im_path, im_size=32, im_channels=3, im_ext='jpg'):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.images = self.load_images(im_path)
    
    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        for fname in glob.glob(os.path.join(im_path, '**/*.{}'.format(self.im_ext)), recursive=True):
            ims.append(fname)
        print('Found {} images'.format(len(ims)))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        im.close()

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor