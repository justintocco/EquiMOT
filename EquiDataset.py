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

        self.dataset = []
        self.annotations = []
        count = 0
        print("Finding annotated frames to train...")
        for img in tqdm(self.root_dir):
            idx_tup = ('set' + str(img[1:3]),'video_' + str(img[7:11]),int(img[13:-4]))
            #print(idx_tup)
            found = False
            if idx_tup[0] in self.pickle_db:
                if idx_tup[1] in self.pickle_db[idx_tup[0]]:
                    if idx_tup[2] in self.pickle_db[idx_tup[0]][idx_tup[1]]:
                        found = True
            #breakpoint() 
            if not found:
                count += 1
            else:
                #breakpoint()
                self.annotations.append(self.pickle_db[idx_tup[0]][idx_tup[1]][idx_tup[2]])
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
        return len(self.dataset)
    
    #Needs updating
    def __getitem__(self, idx):
        img = self.dataset[idx]
        annotation_array = np.zeros((16,10))
        
        for i, object in enumerate(self.annotations[idx]):
            id = object['class']
            if id == 'pedestrian':
                id = 1
            elif id == 'vehicle':
                id = 2
            elif id == 'traffic_light':
                id = 3
            elif id == 'sign':
                id = 4
            elif id == 'crosswalk':
                id = 5
            else:
                id = 0
            annotation_array[i][id] = 1  #One-hot it
            bbox = object['bbox']
            annotation_array[i, 6:10] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            
        annotation = torch.from_numpy(annotation_array)
        if self.transform:
            img = self.transform(img)
        return img, annotation