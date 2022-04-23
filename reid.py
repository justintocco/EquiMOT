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


class ReID(nn.Module):
    def __init__(self):
        # super(Net, self).__init__()
        # self.n_class = N_CLASS

        # TODO change these placeholders to actual variables
        """placeholder tensor for the DLA-34 backbone feature_map;
        input images are 1920x1080; DLA-34 divides dimensions by 4, resulting in 480x270;
        assuming 256 channels for now"""
        self.backbone_map = torch.zeros(256, 270, 480)
        """placeholder tensor for centers variable in detection_branch.py;
        shape (N, 2) where N is the number of objects in the frame"""
        self.centers = torch.zeros(5, 2)
        # placeholder for one-hot ground truth class labels
        self.L = torch.zeros(10, 10)

        # initialize ConvNet layers
        self.conv = nn.Conv2d(self.backbone_map[0], 128, 3, padding=1)
        """TODO the paper doesn't give a lot of specification
        on what this FC layer should look like, adjust as needed"""
        self.fc = nn.Linear(128, 128)
        self.smloss = nn.CrossEntropyLoss()


    def forward(self):
        # initialize relu
        relu = nn.ReLU()

        # convolution layer
        E = self.conv(self.backbone_map)
        E = relu(E)

        # extract re-ID feature vectors from object centers; TODO vectorize this
        for i in range(self.centers.size(0)):
            center = self.centers[i, :]
            c_y = center[1]
            c_x = center[0]
            vec = E[:, c_y, c_x]
            """TODO at this point I'm confused; if softmax outputs a scalar, how is the result a vector P = {p(k), k in [1, K]};
            I may be a little confused as to what is being fed into the fully connected layer and softmax; I believe it has something
            to do with the fact that all objects of the same identity are in the same class - so we perform the FC layer and softmax on EACH
            class of objects individually, then put each one of those softmax loss outputs into the vector P? In that case how do we differentiate
            between which objects are in what class? Must investigate further"""
            vec = self.fc(vec)
            p_k = self.smloss(vec)

        # TODO this is a placeholder because I'm confused what the above for loop is doing - see the TODO directly above
        P = torch.zeros(10)

        L_identity = 0
        # TODO attempt to vectorize this
        for i in range(self.centers.size(0)):
            for k in range(P.size(0)):
                L_identity += self.L[i, k] * torch.log(P[k])

        L_identity = -L_identity
        return L_identity
