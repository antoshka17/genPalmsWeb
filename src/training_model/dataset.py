import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2

class HandDataset(Dataset):
    def __init__(self, images_path, df, transform=None):
        super().__init__()
        self.images_path = images_path
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.iloc[index]['imageName']
        img = cv2.imread(os.path.join(self.images_path, img_name))
        # img = Image.open(os.path.join(self.images_path, img_name)).convert('L')
        # img = img.convert('RGB')
        
        if self.transform is not None:
            img = self.transform(image=img)['image'] / 255.0

        return torch.FloatTensor(img).permute(2, 0, 1)

