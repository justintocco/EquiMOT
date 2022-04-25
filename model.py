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
from scipy.signal import argrelextrema

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

class EquiMOT(nn.Module):
    def __init__(self):
        super(EquiMOT, self).__init__()
        self.weight1 = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.tester = nn.Sequential(
        #     nn.AvgPool2d(10, 10)
        # )
        self.encoder_decoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, stride=4), #STRIDE 4? #1/4
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(4,8,3,padding=1,stride=1), #1
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(8,16,3,padding=1,stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1,stride=1),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32,64,3,padding=(0,1),stride=2,output_padding=1),
            nn.Sigmoid(),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64,128,3,padding=1,stride=2,output_padding=(0,1)),
            nn.Sigmoid(),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128,128,3,padding=1,stride=2,output_padding=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.05)
        )


    def forward(self, x):
        #TODO I believe we might need to pass in targets
        base = self.encoder_decoder(x)
        #base = self.tester(x)
        detection = DetectionBranch()
        reid = ReID()
        detection_output = detection.forward(base)
        np_det = detection_output.detach().numpy()
        np_det = np_det[1][1]

        centers = np.dstack(np.unravel_index(np.argsort(np_det.ravel()), (270, 480)))[0,-16:,:]

        id_output = reid.forward(base,centers)
        output = (detection_output,id_output)
        #output = base # TODO this is temp to get to run
        return output

    def loss(self, outputs, annotations):
        '''
        Loss function implemented inline with FairMOT description of how to
        balance detection and ID branches.
        '''
        #return torch.tensor(0.5, requires_grad=True)
        
        #not sure if labels is needed(left as dumby for now), will explain or look into later
        detection = DetectionBranch()
        reid = ReID()

        detect_loss = 0
        for i, img in enumerate(outputs[0]):
            # Count number of actually annotated items
            num_objects = 0
            for item in annotations[i]:
                if item[8] >= 1:
                    num_objects += 1
            M, feat_centers = detection.heatmap(size=[270,480], gt_boxes=(annotations[i,:,-4:]), N=num_objects, stdev=1)
            #print(img.size())
            detect_loss += detection.heatmap_loss(M=M, M_hat=(img[0]), feat_centers=feat_centers) # TODO params from carlos (indexed from output)
        
        id_loss = 0.5
        '''
        for i, img in enumerate(outputs[1]):
            print(annotations[i,:,:6])
            id_loss = nn.CrossEntropyLoss(img[i], annotations[i,:,:6]) # TODO params from jett (indexed from output)
        '''
        loss = 0.5 * ((1/ np.e**self.weight1)*detect_loss + (1/ np.e**self.weight2)*id_loss+ self.weight1 + self.weight2)
        
        return loss
        

def train(trainloader, net, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    cnt = 0
    net = net.train()
    for images, annotations in tqdm(trainloader):
        images = images.to(device)
        annotations = annotations.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = net.loss(outputs=output, annotations=annotations) #TODO maybe change this area to pass in images, annotations
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        cnt += 1
    end = time.time()
    running_loss /= cnt
    print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
        (epoch, running_loss, end-start))
    return running_loss


def test(testloader, net, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, annotations in testloader:
            images = images.to(device)
            annotations = annotations.to(device)
            output = net(images)
            loss = net.loss(outputs=output, annotations=annotations)
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

