#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time

class final_conv_select_net(nn.Module):
    def __init__(self, use_cuda, K):
        super(final_conv_select_net, self).__init__()
        self.num_classes = K
        self.use_cuda = use_cuda

        # Convolutional layers
        self.conv1 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc = nn.Linear(64, K)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        # Global average pooling
        x = torch.mean(x, dim=2)  # Compute mean across the sequence dimension

        # Apply fully connected layer
        x = self.fc(x)

        # Apply sigmoid activation
        x = self.sigmoid(x)

        return x
    
class feat_extraction_select_net(nn.Module):
    def __init__(self, use_cuda): 
        super(feat_extraction_select_net, self).__init__()
    
        self.use_cuda = use_cuda

        # Initialize network trunks with ResNet pre-trained on ImageNet
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 7 channels
        self.backbone.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final average pooling and fully connected layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add a spatial pooling layer to pool the (2, 2) features
        self.spatial_pooling = nn.AvgPool2d(kernel_size=(2, 2))
        
        # Freeze the parameters of the backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False		

    def forward(self, x):
        # Forward pass through the backbone network
        features = self.backbone(x)
        
        # Apply spatial pooling to the features
        pooled_features = self.spatial_pooling(features)
        
        return pooled_features

class placement_net(nn.Module):
    
    def __init__(self,in_channel=3,out_channel=1):
        super(placement_net, self).__init__()
        self.layer1 = conv_block(in_channel,out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel,out_channel*2)
        self.layer4 = Downsample(out_channel*2)
        self.layer5 = conv_block(out_channel*2,out_channel*4)
        self.layer6 = Downsample(out_channel*4)
        self.layer7 = conv_block(out_channel*4,out_channel*8)
        self.layer8 = Downsample(out_channel*8)
        self.layer9 = conv_block(out_channel*8,out_channel*16)
        self.layer10 = Upsample(out_channel*16)
        self.layer11 = conv_block(out_channel*16,out_channel*8)
        self.layer12 = Upsample(out_channel*8)
        self.layer13 = conv_block(out_channel*8,out_channel*4)
        self.layer14 = Upsample(out_channel*4)
        self.layer15 = conv_block(out_channel*4,out_channel*2)
        self.layer16 = Upsample(out_channel*2)
        self.layer17 = conv_block(out_channel*2,out_channel)
        self.layer18 = nn.Conv2d(out_channel,1,kernel_size=(1,1),stride=1)
        self.act = nn.Sigmoid()

    def forward(self,x):
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x,f4)
        x = self.layer11(x)
        x = self.layer12(x,f3)
        x = self.layer13(x)
        x = self.layer14(x,f2)
        x = self.layer15(x)
        x = self.layer16(x,f1)
        x = self.layer17(x)
        x = self.layer18(x)
        return self.act(x)
    
class Downsample(nn.Module):
    def __init__(self,channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)

class Upsample(nn.Module):
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x,featuremap):
        # x = F.interpolate(x,scale_factor=2,mode='nearest')
        x= F.interpolate(x, size=(featuremap.shape[2], featuremap.shape[3]), mode='nearest')

        x = self.conv1(x)
        print("Shape of x:", x.shape)
        print("Shape of featuremap:", featuremap.shape)       
        x = torch.cat((x,featuremap),dim=1)
        return x

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(3,3),stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1, padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

    