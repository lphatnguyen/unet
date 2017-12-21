import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torchvision
import PIL.Image as Image

class CarvanaDataset(Dataset):
    def __init__(self, csv_path, img_dir, mask_dir, transform=None):
        super(CarvanaDataset, self).__init__()
        self.dataset_csv = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, idx):
        # Image path
        img_name = self.dataset_csv.ix[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        # Mask path
        mask_path = os.path.join(self.mask_dir, img_name.split('.')[0] + '_mask.png')
        img = Image.open(img_path)
        img = img.resize((64,64), Image.BILINEAR)
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        mask = mask.resize((64,64),Image.BILINEAR)
        
        tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img, mask = tensor(img), tensor(mask)
        return img,mask
    


    