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

class final_conv_seq_net(nn.Module):
    def __init__(self, use_cuda, K):
        super(final_conv_seq_net, self).__init__()
        self.use_cuda = use_cuda
        self.num_classes = K

        # Pooling operations along dimensions 3 and 4 (for the 2x2 matrices)
        self.spatial_pooling = nn.AvgPool2d(kernel_size=(2, 2))

        # Pooling operation along dimension 2 (for the 512 dimension)
        self.channel_pooling = nn.AdaptiveAvgPool1d(1)  # Pool to output size 1

        # Fully connected layers
        self.fc1 = nn.Linear(K, 256)
        self.fc2 = nn.Linear(256, K)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        self.score = nn.Sigmoid()

    def forward(self, x):
        # Apply spatial pooling along dimensions 3 and 4
        pooled_features = self.spatial_pooling(x)

        # Reshape to move the channel dimension to dimension 2
        pooled_features = pooled_features.permute(0, 1, 3, 2)

        # Apply channel pooling along dimension 2
        pooled_features = self.channel_pooling(pooled_features)

        # Squeeze to remove singleton dimensions
        pooled_features = pooled_features.squeeze(-1)

        # Apply fully connected layers
        x = self.fc1(pooled_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.score(x)

        return x
    
class feat_extraction_seq_net(nn.Module):

    def __init__(self, use_cuda): 
        super(feat_extraction_seq_net, self).__init__()
    
        self.use_cuda = use_cuda

        # Initialize network trunks with ResNet pre-trained on ImageNet
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
        # Remove the final average pooling and fully connected layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze the parameters of the backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False		

    def forward(self, x):
        # Forward pass through the backbone network
        features = self.backbone(x)
        return features

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
    

    