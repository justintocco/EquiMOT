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

class EquiDataset(Dataset):
    """Custom Dataset for loading into PIE data into EquiMOT."""
    def __init__(self, pkl_file, transform=None):
        print("Initializing Database... this may take a while.")
        super().__init__()
        organize()
        pkl_file = open(pkl_file,'rb')
        self.pickle_db = pickle.load(pkl_file)
        pkl_file.close()
        self.root_name = "processed_images"
        self.root_dir = os.listdir("processed_images")
        self.transform = transform
        #breakpoint()
        print("before dataset")
        self.dataset = []
        count = 0
        print("before for loop")
        for img in self.root_dir:
            idx_tup = ('set' + str(img[1:3]),'video_' + str(img[7:11]),int(img[13:-4]))
            #print(idx_tup)
            found = False
            if idx_tup[0] in self.pickle_db:
                if idx_tup[1] in self.pickle_db[idx_tup[0]]:
                    if idx_tup[2] in self.pickle_db[idx_tup[0]][idx_tup[1]]:
                        found = True
            #breakpoint() 
            if not found:
                #breakpoint()
                #os.remove(os.path.join(self.root_dir, img))
                #print("No Ground Truth for %s",img)
                count += 1
            else:
                #breakpoint()
                self.dataset.append(np.array(Image.open(os.path.join(self.root_name, img))))
        #breakpoint()
        print("Dataset Initiated:")
        print("Annotated frames: ", len(self.dataset))
        print("Deleted frames: ", count)
        #print("Dataset Size: ", len(self.dataset))
        
        
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
        breakpoint()
        return len(self.dataset)
    
    #Needs updating
    def __getitem__(self, idx):
        breakpoint()
        img = torch.FloatTensor(self.dataset[idx]).permute(2, 0, 1)
        annotations = torch.LongTensor(self.q_masks[idx])[None, :, :]
        qmask = self.transform(qmask).squeeze()
        if self.one_hot:
            H, W = qmask.shape
            qmask = torch.nn.functional.one_hot(qmask.reshape(-1), len(self.group2label_idx)).reshape(H, W, -1)
            qmask = qmask.permute(2, 0, 1)
            assert torch.max(qmask) == 1
        return self.transform(img), qmask