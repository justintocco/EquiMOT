import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

""" Model
Design and implement your Convolutional NeuralNetworks to perform semantic segmentation on the MSRC-v2 dataset. 

As mentioned in the course, a normal semantic segmentation model should output a mask with the same size as the input that indicate the pixel-wise classification for the input.

Generally, the basic elements of a convolutional semantic segmentation network include:
1. Convolutional blocks:
    - Typically consists of multiple convolutional layers, activation layers (like ReLU layer), BatchNorm layers, and/or Dropout layers.
2. Down-sampling layers:
    - Can be a simple pooling layer(Max-pooling or Average-Pooling) or convolutional layer with stride not equal to 1
3. Up-sampling layers:
    - Can be a simple up-sampling method like bilinear interpolation or transpose/inversed convolutional layers.
By combining these three types of blocks, you should be able to build your own model to achieve the goal of semantic segmentation.

One example of designing such model inspired by U-Net [1] is:
1. Convolutional block with several Conv-ReLU layers with #channel 3 to C (input to output)
2. Down-sampling layer with down-sampling factor s (output shape (N, C, H/s, W/s))
3. Convolutional block with several Conv-ReLU layers with #channel C to 2C
4. Down-sampling layer with down-sampling factor s (output shape (N, 2C, H/s^2, W/s^2))
5. Single convolutional layer with #channel 2C to 2C
6. Up-sampling layer with up-sampling factor s (output shape (N, 2C, H/s, W/s))
7. Convolutional block with several Conv-ReLU layers with #channel 2C to C
8. Up-sampling layer with up-sampling factor s (output shape (N, C, H, W))
9. Convolutional block with several Conv-ReLU layers with #channel C to N_CLASS
"""

N_CLASS = 5 # will need changing

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        ########################################################################
        # TODO: Implement a sematic segmentation model                         #
        # currently has jake's hw5 code
        ########################################################################
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, stride=1)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(4,32,3,padding=1,stride=1)
        self.conv25 = nn.Conv2d(32,128,3,padding=1,stride=1)
        self.conv3 = nn.Conv2d(128,128,3,padding=1,stride=1)
        self.convTrans05 = nn.ConvTranspose2d(128,32,3,padding=1,stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convTrans1 = nn.ConvTranspose2d(32,4,3,padding=1,stride=1)
        self.convTrans2 = nn.ConvTranspose2d(4,self.n_class,3,padding=1,stride=1)
        self.drop = nn.Dropout(p=0.05)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Implement the forward pass                                     #
        ########################################################################
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.pool(self.relu(self.conv25(x)))
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.relu(self.convTrans05(x))
        x = self.upsample(x)
        x = self.relu(self.convTrans1(x))
        x = self.upsample(x)
        x = self.relu(self.convTrans2(x))
        x = self.drop(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

#maybe move this code to init
# Initialize model
name = 'starter_net'
net = Net().to(device) 
# visualizing the model
print('Your network:')
summary(net, (3,112,112), device=device)



#Train model
def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    cnt = 0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        cnt += 1
    end = time.time()
    running_loss /= cnt
    print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
        (epoch, running_loss, end-start))
    return running_loss

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print('\n',losses / cnt)
    return (losses/cnt)

#left this function in incase we want to visualize the accuracy
def plot_hist(trn_hist, val_hist):
    x = np.arange(len(trn_hist))
    plt.figure(figsize=(12, 8))
    plt.plot(x, trn_hist)
    plt.plot(x, val_hist)
    plt.legend(['Training', 'Validation'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('part3_training_hist.png', dpi=300)
    plt.show()

