import os
import time
import json
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import png
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from PIL import Image
import torchvision
# from colormap.colors import Color, hex2rgb
# from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


# K = # of classes
K = 6

class ReID(nn.Module):
    def __init__(self):
        super().__init__()
        # self.n_class = N_CLASS

        # self.backbone_map = torch.zeros(256, 270, 480)
        # self.centers = torch.zeros(5, 2)
        # self.L = torch.zeros(10, 10)

        # initialize ConvNet layers
        # TODO match input layers to number of channels in backbone_map
        self.conv = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 24, 3, padding=1)
        self.conv4 = nn.Conv2d(24, 6, 3, padding=1)


    """
    "Re-ID branch aims to generate features that can distinguish
    objects. Ideally, affinity among different objects should be
    smaller than that between same objects."

    Inputs:
    backbone_map: the DLA-34 backbone feature map; input images are
                  HxW = 1080x1920. DLA-34 divides dimensions by 4,
                  resulting in 270x480. Assuming 256 channels for now
    centers: the centers variable from detection_branch.py.
             Size (N, 2) where N is the number of objects in the frame.

    Returns:
    P: Size (N, K=6) class distribution vector where K is the number of classes
    """
    def forward(self, backbone_map, centers):
        # initialize relu and softmax
        # TODO use sigmoid instead?
        relu = nn.ReLU()
        softmax = nn.Softmax(dim=1)

        E = self.conv(backbone_map)
        E = relu(E)
        E = self.conv2(E)
        E = relu(E)
        E = self.conv3(E)
        E = relu(E)
        E = self.conv4(E)
        E = relu(E)
        E = softmax(E)

        '''
        # convolution layer
        E = self.conv(backbone_map)
        E = relu(E)

        # N = # of GT objects in frame
        N = centers.shape[0]
        P = torch.zeros(N, K)
        # extract re-ID feature vectors from object centers; TODO vectorize this
        for i in range(N):
            c_x, c_y = centers[i, :]
            print(E.shape)
            vec = E[:,:, c_x, c_y]

            vec = self.fc(vec)
            P[i] = softmax(vec)
        '''

        return E


    """
    Inputs:
    L: Size (N, 10) N x [6 one-hot ground truth class labels +
       4 bounding box coords that this function ignores]
    P: Size (N, K=6) class distribution vector where K is the number of classes
       (calculated from forward() above)

    Returns:
    L_identity: the re-ID loss
    """
    def loss(L, P):
        L_identity = -torch.sum(L[:, 0:6] * torch.log(P))
        return L_identity

        # for k in range(K):
        #     L_identity += L[i, k] * torch.log(P[i, k])

        # for i in range(centers.size(0)):
        #     for k in range(P.size(0)):
        #         L_identity += L[i, k] * torch.log(P[k])
