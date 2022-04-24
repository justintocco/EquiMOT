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
        self.C = 256
        self.first_conv_layer = nn.Conv2d(self.C, 256, 3)
        self.second_conv_layer = nn.Conv2d(256, 256, 1)

    """Each head is implemented by applying a 3 * 3 convolution (with 256 channels) to the output features of DLA-34, 
    followed by a 1 * 1 convolutional layer which generates the final targets." - 4.2 """


    def forward(self, x, annotations):
        #TODO Implement forward pass
        # not sure yet what x should be but general setup
        heatmap, heatmap_loss, centers = self.heatmap(feature_map, size, boxes, N, stdev, M)
        boxes_loss =  self.boxes_loss(boxes, centers, s, o)
        loss = heatmap_loss + boxes_loss
        return x, loss, centers


    def heatmap(feature_map, size, boxes, N, stdev, M):
        """
        Responsible for estimating the locations of the object centers
        "The response at a location in the heatmap is expected to be one if it collapses
        with the ground-truth object center."

        Inputs:
        feature_map - the output of the DLA-34 model
        size - a tuple of ints giving the (height, width) of original image / 4
        N - the number of objects in the image
        boxes - an (N, 4) array of ground truth bounding boxes in the image 
        stdev - standard deviation (TODO: is this passed in or is it calculated?)
        M - real heatmap responses

        Notes:
        Height of the feature map is the same as the height of the heatmap, which
        is the height of the original image / 4. The same applies for the width.

        Returns:
        M_hat: (1 * H * W).
        L_heat: the heatmap loss
        """
        STRIDE = 4 

        M_hat = torch.zeros((1, size[0], size[1]))
        centers = torch.zeros((N, 2))
        L_heat = None

        #for each box in the image
        for i in range(N): #TODO: try to make this w/o loops
            #compute the object center
            obj_center = torch.array((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
            #location on the feature map is obtained by dividing the stride (c~i_x, c~i_y)
            on_feat_map = torch.floor(obj_center / STRIDE)
            centers[i] = obj_center #center of ith box of object

                
        x_range = torch.arange(0, size[0], 1)
        y_range = torch.arange(0, size[1], 1)
        response = 0
        #there's totally a more efficient way to do this
        for x in range(x_range):
            for y in range(y_range):
                for i in range(N):
                    numerator = ((x - centers[i][0]) ** 2) + ((y - centers[i][1]) ** 2)
                    denominator = 2 * (stdev ** 2)
                    term = -(numerator/denominator)
                    term = torch.exp(term)
                    response+=term
                M_hat[x][y] = response
                response = 0

        
        #heatmap_est = torch.sum( torch.exp(- ((x - c_x) ** 2) + ((y - c_y) ** 2) / (2 * (stdev ** 2))) )

        #alpha and beta are the pre-determined parameters in focal loss
        alpha = 1
        beta = 1
        #loss function is defined as pixel-wise logistic regression with focal loss
        loss_arrs = torch.zeros((M.shape))
        loss_arrs[M == 1] = ((1 - M_hat) ** alpha) * np.log(M_hat)
        loss_arrs[M != 1] = ((1 - M) ** beta) * (M_hat ** alpha) * np.log(1 - M_hat)
        L_heat = torch.sum(loss_arrs)
        
        return heatmap_est, L_heat, centers

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

    def boxes_loss(boxes, centers, s, o):
        s_hat = size(boxes)
        o_hat = box_offset(centers)
        lamba = 0.1 #set to this in the original CenterNet
        #o is GT offset
        #TODO: can you use numpy in tensor operations?
        L_box = torch.sum(np.linalg.norm(o - o_hat, ord=1) + (lamba * np.linalg.norm(s - s_hat, ord=1)))
        return L_box




    


