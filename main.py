import os
import numpy as np
import torch
import torch.nn as nn
import model
import data_loader as dl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from EquiDataset import EquiDataset

if torch.cuda.is_available():
    print("Using the GPU.")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed will be slowed down.")
    device = 'cpu'

name = 'starter_net'
net = model.Net().to(device)
criterion = nn.CrossEntropyLoss() # this will be the loss we create using the parallel heads for tracking and detection
#breakpoint()
# Define the dataset and dataloder
dataset = EquiDataset(pkl_file = 'small_database.pkl',transform=transforms.ToTensor())
train_data = None
val_data =  None
test_data = None

train_loader = DataLoader(train_data, batch_size=16)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=1)

optimizer = model.optim.Adam(net.parameters(), lr=2e-4,weight_decay=1e-5) 
num_epoch = 80

print('\nStart training')
trn_hist = []
val_hist = []
net.train()
for epoch in range(num_epoch): #TODO: Change the number of epochs
    print('-----------------Epoch = %d-----------------' % (epoch+1))
    trn_loss = model.train(train_loader, net, criterion, optimizer, device, epoch+1)
    print('Validation loss: ')
    val_loss = model.test(val_loader, net, criterion, device)
    trn_hist.append(trn_loss)
    val_hist.append(val_loss)

net.eval()
model.plot_hist(trn_hist, val_hist)
print('\nFinished Training, Testing on test set')
model.test(test_loader, net, criterion, device)
print('\nGenerating Unlabeled Result')

#TODO not sure what these two lines do yet lol (some sort of saving the model)
os.makedirs('./models', exist_ok=True)
torch.save(net.state_dict(), './models/model_skiplink_{}.pth'.format(name))
