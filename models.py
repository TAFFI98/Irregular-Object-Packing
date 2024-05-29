# -*- coding: utf-8 -*-
"""
Created on  Apr 3 2024

@author: Taffi
"""
import pybullet as p
import numpy as np
import torch
from env import Env
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import torch.nn.init as init
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
def even_odd_sign(n):
    if n % 2 == 0:
        return 1
    else:
        return -1
    
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
        # Set to zero the fetures of objects already packed
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
    def __init__(self, n_y, in_channel_unet = 3, out_channel = 1, use_cuda= False):
        super(placement_net, self).__init__()
        self.n_y = n_y
        self.use_cuda = use_cuda

        # Define layers for U-Net architecture
        self.layer1_conv_block = conv_block(in_channel_unet, out_channel)
        self.layer2_Downsample = Downsample(out_channel)
        self.layer3_conv_block = conv_block(out_channel, out_channel*2)
        self.layer4_Downsample = Downsample(out_channel*2)
        self.layer5_conv_block = conv_block(out_channel*2, out_channel*4)
        self.layer6_Downsample = Downsample(out_channel*4)
        self.layer7_conv_block = conv_block(out_channel*4, out_channel*8)
        self.layer8_Downsample = Downsample(out_channel*8)
        self.layer9_conv_block = conv_block(out_channel*8, out_channel*16)
        self.layer10_Upsample = Upsample(out_channel*16)
        self.layer11_conv_block = conv_block(out_channel*16, out_channel*8)
        self.layer12_Upsample = Upsample(out_channel*8)
        self.layer13_conv_block = conv_block(out_channel*8, out_channel*4)
        self.layer14_Upsample = Upsample(out_channel*4)
        self.layer15_conv_block = conv_block(out_channel*4, out_channel*2)
        self.layer16_Upsample = Upsample(out_channel*2)
        self.layer17_conv_block = conv_block(out_channel*2, out_channel)
        self.layer18 = nn.Conv2d(out_channel, 1, kernel_size=(1, 1), stride=1) 
        # Without an activation function, the output values of the heatmap produced by the final convolutional 
        # layer can assume any real number, ranging from negative to positive infinity.

    def forward(self, env, roll_pitch_angles, input1, input2):
        batch_size = input1.size(0)
        n_rp = input1.size(1)

        # Reshape input2 to match the batch size of input1
        input2 = input2.expand(batch_size, -1, -1, -1)
        
        # Apply rotations to input1
        rows = []
        for j in range(n_rp):
            input1_rp = input1[:,j,:,:,:] #[batch,res,res,2]
            score_matrices = []           #[batch,res,res,3,ny]
            for i in range(self.n_y): 
                angle =  i * (360 / self.n_y)
                orient = list(roll_pitch_angles[j]) + [angle]
                #print('Worker network evaluating orientation: ', orient)
                unet_input = self.rotate_tensor_and_append_bbox(input1_rp, orient, input2, batch_size, i, env) #[batch,3,res,res]
                unet_output = self.unet_forward(unet_input)
                score_matrices.append(unet_output)# list of score matrices [batch,res,res]-- one for each yaw
            
            rows.append(torch.stack(score_matrices, dim=1)) #[batch, n_rp, res, res]    
        
        Q_values = torch.stack(rows, dim=1) #[batch, n_y, n_rp, res, res]
 
        return Q_values

    def unet_forward(self, x):
        ''' Downsample'''
        x = self.layer1_conv_block(x)
        f1 = x
        x = self.layer2_Downsample(x)
        x = self.layer3_conv_block(x)
        f2 = x
        x = self.layer4_Downsample(x)
        x = self.layer5_conv_block(x)
        f3 = x
        x = self.layer6_Downsample(x)
        x = self.layer7_conv_block(x)
        f4 = x
        ''' Upsample'''
        x = self.layer8_Downsample(x)
        x = self.layer9_conv_block(x)
        x = self.layer10_Upsample(x, f4)
        x = self.layer11_conv_block(x)
        x = self.layer12_Upsample(x, f3)
        x = self.layer13_conv_block(x)
        x = self.layer14_Upsample(x, f2)
        x = self.layer15_conv_block(x)
        x = self.layer16_Upsample(x, f1)
        x = self.layer17_conv_block(x)

        x = self.layer18(x)
        return torch.squeeze(x, dim=1) #[batch,res,res]
    
    def rotate_tensor_and_append_bbox(self, input1, orient, input2, batch_size, i , env):
 
        # Generate rotation matrix
        angle = orient[2] 
        rotation_matrix = np.asarray([[even_odd_sign(i)* np.cos(angle),even_odd_sign(i)* -np.sin(angle), 0],[even_odd_sign(i)*np.sin(angle), even_odd_sign(i)*np.cos(angle), 0]])
        rotation_matrix.shape = (2,3,1)
        rotation_matrix = torch.from_numpy(rotation_matrix).permute(2,0,1).float() #[1,2,3]
        rotation_matrix = rotation_matrix.repeat(batch_size, 1, 1)
        # put channels before in the input 1 
        input1 = input1.permute(0,3,1,2).float() # [batch,2,res,res]
        input2 = input2.permute(0,3,1,2).float() # [batch,1,res,res]
        if self.use_cuda == True :
                    flow_grid_before = F.affine_grid(Variable(rotation_matrix, requires_grad=False).cuda(), input1.size(),align_corners=True)
                    rotated_hm = F.grid_sample(Variable(input1, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True) #[batch,2,res,res]
                    rotated_hm = rotated_hm.cuda()
                    input2 = input2.cuda()
        elif self.use_cuda == False:
                    flow_grid_before = F.affine_grid(Variable(rotation_matrix, requires_grad=False), input1.size(),align_corners=True)
                    rotated_hm = F.grid_sample(Variable(input1, requires_grad=False), flow_grid_before, mode='nearest',align_corners=True) #[batch,2,res,res]
            
        # -----> Uncomment to Show the rotated heightmaps
        # env.visualize_object_heightmaps(input1[0,0,:,:].detach().cpu().numpy(), input1[0,1,:,:].detach().cpu().numpy(), [orient[0],orient[1],0], only_top = False)
        # env.visualize_object_heightmaps(rotated_hm[0,0,:,:].detach().cpu().numpy(), rotated_hm[0,1,:,:].detach().cpu().numpy(), orient, only_top = False)
        # env.visualize_object_heightmaps(input2[0, 0, :, :].detach().cpu().numpy(), None, [0,0,0], only_top = True)
        
        # Concatenate the rotated heightmap with the box heightmap
        input_unet = torch.cat([rotated_hm,input2], dim=1)

        return input_unet

class Downsample(nn.Module):
    '''
    The Downsample class is responsible for reducing the spatial dimensions of the feature maps by half while maintaining the number of channels.
    '''
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
    '''
    The Upsample class is responsible for increasing the spatial dimensions of the feature maps by a factor of 2 and concatenating it with the corresponding feature map from the downsampling path along the channel direction.
    '''
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x,featuremap):
        x= F.interpolate(x, size=(featuremap.shape[2], featuremap.shape[3]), mode='nearest')
        x = self.conv1(x)
        x = torch.cat((x,featuremap),dim=1)
        return x

class conv_block(nn.Module):
    '''
    The conv_block class encapsulates two sequential convolutional operations, each followed by batch normalization, dropout, and ReLU activation.
    '''
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
        self.apply(self.init_weights)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #init.xavier_uniform_(m.weight)
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    
