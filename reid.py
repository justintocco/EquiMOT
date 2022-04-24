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


# K = # of classes TODO confirm this is 5?
K = 5

class ReID(nn.Module):
    def __init__(self):
        # super(Net, self).__init__()
        # self.n_class = N_CLASS

        # self.backbone_map = torch.zeros(256, 270, 480)
        # self.centers = torch.zeros(5, 2)
        # self.L = torch.zeros(10, 10)

        # initialize ConvNet layers
        # TODO match input layers to number of channels in backbone_map
        self.conv = nn.Conv2d(256, 128, 3, padding=1)
        self.fc = nn.Linear(128, K)


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
    L: Size (N, K=5) one-hot ground truth class labels

    Returns:
    P: Size (N, K=5) class distribution vector where K is the number of classes
    L_identity: the re-ID loss
    """
    def forward(self, backbone_map, centers, L):
        # initialize relu and softmax
        # TODO use sigmoid instead?
        relu = nn.ReLU()
        softmax = nn.Softmax(dim=0)

        # convolution layer
        E = self.conv(backbone_map)
        E = relu(E)

        # N = # of GT objects in frame
        N = centers.size(0)
        P = torch.zeros(N, K)
        L_identity = 0
        # extract re-ID feature vectors from object centers; TODO vectorize this
        for i in range(N):
            c_x, c_y = centers[i, :]
            vec = E[:, c_y, c_x]

            vec = self.fc(vec)
            P[i] = softmax(vec)
            L_identity += torch.sum(L[i, :] * torch.log(P[i, :]))
            # for k in range(K):
            #     L_identity += L[i, k] * torch.log(P[i, k])

        # for i in range(centers.size(0)):
        #     for k in range(P.size(0)):
        #         L_identity += L[i, k] * torch.log(P[k])

        L_identity = -L_identity
        return P, L_identity
