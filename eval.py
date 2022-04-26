import os
import numpy as np
import torch
import torch.nn as nn
import model
import data_loader as dl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from EvalDataset import EvalDataset
from torchsummary import summary
from PIL import Image

if torch.cuda.is_available():
    print("Using the GPU.")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed will be slowed down.")
    device = 'cpu'

name = 'starter_net'
# Define the dataset and dataloder
dataset = EvalDataset(transform=transforms.ToTensor())
#train_data, val_data, test_data = torch.utils.data.random_split(dataset, [200,200,3075]) # this may need changing

#print(dataset)
#print(test_data)

#train_loader = DataLoader(train_data, batch_size=8)
#val_loader = DataLoader(val_data, batch_size=8)
#test_loader = DataLoader(test_data)

net = model.EquiMOT().to(device)
net.load_state_dict(torch.load('models/model_skiplink_starter_net.pth'))
summary(net, (3,1080,1920), device=device)

print("MODEL PARAMETERS:")
print()
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)

print("---------------------------------------------------------")
eval_data = DataLoader(dataset)

#optimizer = model.optim.Adam(net.parameters(), lr=1e-2,weight_decay=1e-5) 
#num_epoch = 12
#print('\nStart training')

result = model.get_result(eval_data, net, device, folder='output_test')

