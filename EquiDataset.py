import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import json

class EquiDataset(Dataset):
    """Custom Dataset for loading into PIE data into EquiMOT."""
    def __init__(self, 
                file,root_dir,transform=None):
        super().__init__()
        
        self.dataset = [np.array(Image.open(os.path.join(root_dir, img))) 
                     for img in self.img_list]
        
       
        """
        self.img_list = [data[0] for data in self.dataset]
        self.mask_list = [data[1] for data in self.dataset]
        GTQ_list = [data[2] for data in self.dataset]
        self.imgs = [np.array(Image.open(os.path.join(img_folder, img))) 
                     for img in self.img_list]
        self.masks = [np.array(Image.open(os.path.join(mask_folder, mask))) 
                      for mask in self.mask_list]
        self.q_masks = [np.load(os.path.join(GTQ_folder, gtq))
                        for gtq in GTQ_list]
            
        self.transform = transforms.Compose([
            transforms.Resize((112, 112))
        ])
        """

        
    def __len__(self):
        return len(self.dataset)
    
    #Needs updating
    def __getitem__(self, idx):
        img = torch.FloatTensor(self.imgs[idx]).permute(2, 0, 1)
        qmask = torch.LongTensor(self.q_masks[idx])[None, :, :]
        qmask = self.transform(qmask).squeeze()
        if self.one_hot:
            H, W = qmask.shape
            qmask = torch.nn.functional.one_hot(qmask.reshape(-1), len(self.group2label_idx)).reshape(H, W, -1)
            qmask = qmask.permute(2, 0, 1)
            assert torch.max(qmask) == 1
        return self.transform(img), qmask