import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import json
import pickle
from data_organizer import organize
from tqdm import tqdm


class EvalDataset(Dataset):
    """Custom Dataset for loading unannotated images into EquiMOT."""
    def __init__(self, transform=None):
        print("Initializing Database...")
        super().__init__()
        organize()
        
        self.root_name = "processed_images"
        self.root_dir = os.listdir("processed_images")
        self.transform = transform
        #breakpoint()

        self.dataset = []
        self.annotations = []
        count = 0
        print("Finding evaluation frames...")
        for frame in range(2970, 3460, 10):
            img = 's03_vid0003_f' + str(frame) + '.png'
            self.dataset.append(np.array(Image.open(os.path.join(self.root_name, img))))

        print("Dataset Initiated:")
        print("Eval frames: ", len(self.dataset))
        
        
    def __len__(self):
        return len(self.dataset)
    
    #Needs updating
    def __getitem__(self, idx):
        img = self.dataset[idx]
        annotation = np.zeros((16,10))
    
        if self.transform:
            img = self.transform(img)
        return img, annotation