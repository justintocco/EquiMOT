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
from detection_branch import DetectionBranch
from reid import ReID

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

N_CLASS = 5 # MAY (not) need changing

class EquiMOT(nn.Module):
    def __init__(self):
        super(EquiMOT, self).__init__()
        self.n_class = N_CLASS
        self.encoder_decoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(4,32,3,padding=1,stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,128,3,padding=1,stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(128,128,3,padding=1,stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128,32,3,padding=0,stride=2),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32,8,3,padding=0,stride=2),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(8,self.n_class,3,padding=0,stride=2),
            nn.Sigmoid(),
            nn.Dropout(p=0.05)
        )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x, annotations):
        #TODO I believe we might need to pass in targets
        base = self.encoder_decoder(x)
        detection = DetectionBranch()
        reid = ReID()
        detection_output, detection_loss, centers = detection.forward(base,annotations)
        id_output, id_loss = reid.forward(base,centers,annotations)
        branch_outputs = (detection_output,id_output)
        branch_losses = (detection_loss,id_loss)
        return (branch_losses,branch_outputs)


def loss(self, outputs, labels=None):
    '''
    Loss function implemented inline with FairMOT description of how to
    balance detection and ID branches.
    '''
    #not sure if labels is needed(left as dumby for now), will explain or look into later
    
    w1, w2 = None, None # TODO need to figure out how to implement learnable parameters

    #outputs[0][0] refers to detecion loss, output[0][1] refers to id_loss
    loss = 0.5 * ((1/ np.e**w1)*outputs[0][0] + (1/ np.e**w2)*outputs[0][1]+ w1 + w2)
    return loss

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    cnt = 0
    net = net.train()
    for images, labels, bboxes in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels) #TODO maybe change this area to pass in images, annotations
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
        for images, labels in testloader:
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

