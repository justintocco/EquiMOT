import time
import threading
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
import math

"""
"Our detection branch is built on top of CenterNet (Zhou et al., 2019a) but other 
anchor-free methods such as (Duan et al., 2019; Law and Deng, 2018; Dong et al., 2020; 
Yang et al., 2019) can also be used. We briefly describe the ap- proach to make this work self-contained. 
In particular, three parallel heads are appended to DLA-34 to estimate heatmaps, 
object center offsets and bounding box sizes, respectively. 
"""

"""
The output size of DLA-34 is (C * H * W), where:
H = height of original image / 4
W = width of original image / 4

This result is what feeds into the detection (as well as Re-ID) branch.
"""

class DetectionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 128
        self.relu = nn.ReLU()
        self.first_conv_layer = nn.Conv2d(self.C, 64, 3, padding=1)
        self.second_conv_layer = nn.Conv2d(64, 16, 3, padding=1)
        self.third_conv_layer = nn.Conv2d(16, 5, 1)


    def forward(self, x):
        #TODO Implement forward pass
        # not sure yet what x should be but general setup
        x = self.first_conv_layer(x)
        x = self.relu(x)
        x = self.second_conv_layer(x)
        x = self.relu(x)
        x = self.third_conv_layer(x)
        x = self.relu(x)
        #heatmap, heatmap_loss, centers = self.heatmap(feature_map, size, boxes, N, stdev, M)
        #boxes_loss =  self.boxes_loss(boxes, centers, s, o)
        #loss = heatmap_loss + boxes_loss
        return x


    def heatmap(self, size, gt_boxes, N, stdev):
        """
        Responsible for estimating the locations of the object centers
        "The response at a location in the heatmap is expected to be one if it collapses
        with the ground-truth object center."

        Inputs:
        size - a tuple of ints giving the (height, width) of original image
        N - the number of objects in the image
        gt_boxes - an (N, 4) array of ground truth bounding boxes in the image 
        stdev - standard deviation (TODO: call stdev on the input tensor and get it passed here)

        Notes:
        Height of the feature map is the same as the height of the heatmap, which
        is the height of the original image / 4. The same applies for the width.

        Returns:
        M: (1 * H * W).
        """
        STRIDE = 4 

        #M_hat = torch.zeros((1, size[0], size[1]))
        centers = torch.zeros((N, 2))
        L_heat = None

        #given GT boxes:
        #compute GT center
        #divide GT center by stride
        #use this to this M

        feat_centers = []
        #print(gt_boxes)

        #for each box in the image
        for i in range(N): #TODO: try to make this w/o loops
            #compute the object center
            gt_obj_center = [(gt_boxes[i][1] + gt_boxes[i][3]) / 2.0, (gt_boxes[i][0] + gt_boxes[i][2]) / 2.0]
            #print(gt_obj_center)
            #location on the feature map is obtained by dividing the stride (c~i_x, c~i_y)
            on_feat_map = [int(gt_obj_center[0]/STRIDE), int(gt_obj_center[1]/STRIDE)]
            #print(on_feat_map)
            feat_centers.append(on_feat_map) #center of ith box of object

        #print("FEAT Centers:")
        #print(feat_centers)
        M = torch.zeros(size)
        #x_range = np.arange(0, size[0], 1)
        #y_range = np.arange(0, size[1], 1)
        response = 0
        #TODO: there's totally a more efficient way to do this
        for x in range(size[0]):
            for y in range(size[1]):
                for i in range(N):
                    numerator = ((x - feat_centers[i][0]) ** 2) + ((y - feat_centers[i][1]) ** 2)
                    denominator = 2 * (stdev ** 2)
                    term = -(numerator/denominator)
                    term = np.e ** term
                    response += term
                M[x][y] = response
                response = 0

        return M, feat_centers

    def size(boxes):
        """
        Predict size of box at each location

        Output: S_hat in the set of R^(2 * H * W)
        """
        box_sizes = torch.array(boxes.shape[0], 2)
        for i in range(boxes.shape[0]):
            box_sizes[i] = (boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]) 
        return box_sizes

    def box_offset(centers):
        """
        Output: S_hat in the set of R^(2 * H * W)
        """
        offset = torch.array(centers.shape[0], 1)
        offset = center/4 - np.floor(center/4)

        return offset

    def heatmap_loss(self, M, M_hat, feat_centers):
        """
        Inputs:
        M - GT
        M_hat - Predicted heatmap
        """
        #print("M")
        #print(M.size())
        #print("M-hat")
        #print(M_hat.size())
        #alpha and beta are the pre-determined parameters in focal loss
        alpha = 1
        beta = 1
        #loss function is defined as pixel-wise logistic regression with focal loss
        #loss_arrs = torch.zeros(size=M.size())
        #loss_arrs[M == 1] = ((1 - M_hat) ** alpha) * torch.log(M_hat)
        #loss_arrs[M != 1] = ((1 - M) ** beta) * (M_hat ** alpha) * torch.log(1 - M_hat)

        #print(feat_centers)
        loss_arr = ((1 - M) ** beta) * (M_hat ** alpha) * torch.log(1 - M_hat)
        for cen in feat_centers:
            x = cen[0]
            y = cen[1]
            loss_arr[x][y] = ((1 - M_hat[x][y]) ** alpha) * math.log(M_hat[x][y]+0.01)

        L_heat = torch.sum(loss_arr)
        L_heat = (L_heat * -1) / len(feat_centers)
        return L_heat

    def boxes_loss(boxes, centers, s, o):
        s_hat = size(boxes)
        o_hat = box_offset(centers)
        lamba = 0.1 #set to this in the original CenterNet
        #o is GT offset
        #TODO: can you use numpy in tensor operations?
        L_box = torch.sum(np.linalg.norm(o - o_hat, ord=1) + (lamba * np.linalg.norm(s - s_hat, ord=1)))
        return L_box
