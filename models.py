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
''' 
selection_net and placement_net are the classes to be imported
from models import  placement_net, selection_net

SELECTION NET
input1_selection_network ---> (batch, K, 6, res, res) - 6 views of the K objects​
input2_selection_network ---> (batch, 1, res, res) - BOX Heightmap​
manager_network = selection_net(use_cuda = False,K = K)
chosen_item_index, score_values = manager_network(torch.tensor(input1_selection_network),torch.tensor(input2_selection_network),item_ids)
score_values ---> (batch, K) - Scores for the objects representing the probability of being the next obj to be paked​
chosen_item_index ---> int  - Index of the next object to be packed

PLACEMENT NET
​input_placement_network_1 ---> (batch, n_rp, res, res, 2) - top and bottom HM of the selected objects at n_rp orientations​
input_placement_network_2 ---> (batch, res, res, 1) - BOX Heightmap
worker_network = placement_net(use_cuda = False)
Q_values , [pixel_x,pixel_y,r,p,y] = worker_network(torch.tensor(input_placement_network_1),torch.tensor(input_placement_network_2),roll_pitch_angles)
[pixel_x,pixel_y,r,p,y] ---> selected pose​
Q_values ---> (batch, n_y, n_rp, res, res) - Q values for the discretized actions


'''  

class selection_net(nn.Module):
    def __init__(self, use_cuda, K): 
        super(selection_net, self).__init__()
    
        self.use_cuda = use_cuda
        self.K = K
        
        # Initialize network trunks with ResNet pre-trained on ImageNet
        self.backbones = nn.ModuleList([self.initialize_backbone() for _ in range(self.K)])
        
        # Add a spatial pooling layer to pool the (2, 2) features
        self.spatial_pooling = nn.AvgPool2d(kernel_size=(2, 2))
        
        # Freeze the parameters of the backbone networks
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = False		

    def initialize_backbone(self):
        # Initialize ResNet backbone
        backbone = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 7 channels
        backbone.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final average pooling and fully connected layers
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        return backbone

    def forward(self, input_1, input_2, item_ids):


        concatenated_features,k_already_packed = [],[]
        for k in range(self.K):
            # Concatenate input_1 with input_2 along dimension=-3
            if not torch.any(input_1[:, k, :, :, :][input_1[:, k, :, :, :]!= 0]):
                k_already_packed.append(k)
                concatenated_input = torch.cat((input_1[:, k, :, :, :], input_2), dim=1)
                # Forward pass through the backbone network
                backbone_output = self.backbones[k](concatenated_input.float())#(batch,512,2,2)
            else:
                concatenated_input = torch.cat((input_1[:, k, :, :, :], input_2), dim=1)
                # Forward pass through the backbone network
                backbone_output = self.backbones[k](concatenated_input.float())#(batch,512,2,2)
            
            # Apply spatial pooling to the features
            pooled_features = self.spatial_pooling(backbone_output)           #(batch,512,1,1)
            pooled_features = pooled_features.squeeze(dim=-1).squeeze(dim=-1) #(batch,512)

            concatenated_features.append(pooled_features) 
        
        # Concatenate the features from all backbones
        concatenated_features = torch.stack(concatenated_features, dim=1)      #(batch,K,512)
        final_selection_layers = final_conv_select_net(self.use_cuda,self.K)
        score_values = final_selection_layers(concatenated_features)            #(batch,K)
        for kk in k_already_packed:
            score_values[:,kk] = 0.0
        
        chosen_item_index = torch.argmax(score_values, dim=1)
        chosen_item_index_pybullet = item_ids[chosen_item_index]
        return chosen_item_index_pybullet, score_values

class final_conv_select_net(nn.Module):
    def __init__(self, use_cuda, K):
        super(final_conv_select_net, self).__init__()
        self.num_classes = K
        self.use_cuda = use_cuda

        # Define fully connected layers for each branch
        self.fc_layers = nn.ModuleList([self.create_fc_layers() for _ in range(K)])
        
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def create_fc_layers(self):
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        outputs = []
        # Loop through each branch
        for k in range(self.num_classes):
            # Get the features for the k-th branch
            x_k = x[:, k, :]  # Extract features for the k-th class (batch, 512)
            # Forward pass through the fully connected layers for this branch
            output_k = self.fc_layers[k](x_k)  # (batch, 1)
            outputs.append(output_k)

        # Concatenate the outputs along the class dimension
        concatenated_outputs = torch.cat(outputs, dim=1)  # (batch, K)

        # Apply sigmoid activation
        concatenated_outputs = self.sigmoid(concatenated_outputs) # (batch, K)

        return concatenated_outputs
    
class placement_net(nn.Module):
    def __init__(self, in_channel_unet=3, out_channel=1, n_y=16, use_cuda= False):
        super(placement_net, self).__init__()
        self.n_y = n_y
        self.use_cuda = use_cuda

        # Define layers for U-Net architecture
        self.layer1 = conv_block(in_channel_unet, out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel, out_channel*2)
        self.layer4 = Downsample(out_channel*2)
        self.layer5 = conv_block(out_channel*2, out_channel*4)
        self.layer6 = Downsample(out_channel*4)
        self.layer7 = conv_block(out_channel*4, out_channel*8)
        self.layer8 = Downsample(out_channel*8)
        self.layer9 = conv_block(out_channel*8, out_channel*16)
        self.layer10 = Upsample(out_channel*16)
        self.layer11 = conv_block(out_channel*16, out_channel*8)
        self.layer12 = Upsample(out_channel*8)
        self.layer13 = conv_block(out_channel*8, out_channel*4)
        self.layer14 = Upsample(out_channel*4)
        self.layer15 = conv_block(out_channel*4, out_channel*2)
        self.layer16 = Upsample(out_channel*2)
        self.layer17 = conv_block(out_channel*2, out_channel)
        self.layer18 = nn.Conv2d(out_channel, 1, kernel_size=(1, 1), stride=1)

    def forward(self, input1, input2, roll_pitch_angles):
        batch_size = input1.size(0)
        resolution = input1.size(2)
        n_rp = input1.size(1)

        # Reshape input2 to match the batch size of input1
        input2 = input2.expand(batch_size, -1, -1, -1)
        
        # Apply rotations to input1
        rows = []
        for i in range(n_rp):
            input1_rp = input1[:,i,:,:,:] #[batch,res,res,2]
            rotated_inputs = [] #[batch,res,res,3,ny]
            for i in range(self.n_y):
                angle = i * (360 / self.n_y)
                unet_input = self.rotate_tensor_and_append_bbox(input1_rp, angle, input2) #[batch,3,res,res]
                unet_output = self.unet_forward(unet_input)
                rotated_inputs.append(unet_output)#[batch,res,res]
            
            rows.append(torch.stack(rotated_inputs, dim=1)) #[batch, n_rp, res, res]
        
        Q_values = torch.stack(rows, dim=1) #[batch, n_y, n_rp, res, res]
 

        # Get the maximum  element and its indices along each dimension
        max_index_flat = torch.argmax(Q_values.view(-1))
        max_index = torch.unravel_index(max_index_flat, Q_values.size())


        # Extract indices along each dimension
        indices_y = int(max_index[1])
        indices_rp = int(max_index[2])
        pixel_x = int(max_index[3])
        pixel_y = int(max_index[4])
        

        r = float(roll_pitch_angles[indices_rp,0])
        p = float(roll_pitch_angles[indices_rp,1])
        y = float(indices_y * (360 / self.n_y))

        return Q_values, [pixel_x,pixel_y,r,p,y]

    def unet_forward(self, x):
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
        x = self.layer10(x, f4)
        x = self.layer11(x)
        x = self.layer12(x, f3)
        x = self.layer13(x)
        x = self.layer14(x, f2)
        x = self.layer15(x)
        x = self.layer16(x, f1)
        x = self.layer17(x)
        x = self.layer18(x)
        return torch.squeeze(x, dim=1)
    
    def rotate_tensor_and_append_bbox(self, input1, angle, input2):
 
        # Generate rotation matrix
        rotation_matrix = np.asarray([[np.cos(-angle), np.sin(-angle), 0],[-np.sin(-angle), np.cos(-angle), 0]])
        rotation_matrix.shape = (2,3,1)
        rotation_matrix = torch.from_numpy(rotation_matrix).permute(2,0,1).float() #[1,2,3]
        
        # put channels before in the input 1 
        input1 = input1.permute(0,3,1,2).float()#[batch,2,res,res]
        input2 = input2.permute(0,3,1,2).float() #[batch,1,res,res]
        if self.use_cuda == True :
                    flow_grid_before = F.affine_grid(Variable(rotation_matrix, requires_grad=False).cuda(), input1.size(),align_corners=True)
                    rotated_hm = F.grid_sample(Variable(input1, requires_grad=False).cuda(), flow_grid_before, mode='nearest',align_corners=True) #[batch,2,res,res]
        elif self.use_cuda == False:
                    flow_grid_before = F.affine_grid(Variable(rotation_matrix, requires_grad=False), input1.size(),align_corners=True)
                    rotated_hm = F.grid_sample(Variable(input1, requires_grad=False), flow_grid_before, mode='nearest',align_corners=True) #[batch,2,res,res]

        input_unet = torch.cat([rotated_hm,input2], dim=1)

        
        return input_unet

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
        x= F.interpolate(x, size=(featuremap.shape[2], featuremap.shape[3]), mode='nearest')
        x = self.conv1(x)
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

    
