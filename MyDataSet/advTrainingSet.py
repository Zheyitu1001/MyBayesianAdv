import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.init as init

import torchvision.transforms as transforms

import numpy as np
import cv2, os
from PIL import Image

class advData(Dataset):
    def __init__(self, root_path, transform=None):
        super(advData, self).__init__()
        data_list = os.listdir(root_path)
        imgs = []
        for item in data_list:
            img_path = os.path.join(root_path, item)
            labels = item.split('.')[0]
            labels = labels.split('_')[1]
            imgs.append((img_path, int(labels)))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img, label = self.imgs[index]
        image = Image.open(img)
        #image = cv2.imread(img)
        #image = np.transpose(image, (2, 0, 1))
        #image = image.astype(np.float32)
        #b, g, r = cv2.split(image)
        #image = cv2.merge([r, g, b])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.imgs)