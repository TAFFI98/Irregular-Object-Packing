# -*- coding: utf-8 -*-
"""
Created on  Apr 3 2024

@author: Taffi
"""
# from torchview import draw_graph
from fonts_terminal import *
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
import random
import os
''' 
The selection_placement_net class is responsible for selecting the best object and placing it in the scene.
The selection_placement_net class is composed of two sub-networks: the selection network and the placement network.
The selection network is responsible for selecting the best object from a list of K objects.
The placement network is responsible for placing the selected object in the scene.
The selection_placement_net class is composed of the following methods:


The Downsample class is responsible for reducing the spatial dimensions of the feature maps by half while maintaining the number of channels.   
The Upsample class is responsible for increasing the spatial dimensions of the feature maps by a factor of 2 and concatenating it with the corresponding feature map from the downsampling path along the channel direction.
The conv_block class encapsulates two sequential convolutional operations, each followed by batch normalization, dropout, and ReLU activation.
The rotate_tensor_and_append_bbox method rotates the input tensor and appends the bounding box heightmap to the rotated heightmap.
The even_odd_sign method returns 1 if n is even and -1 if n is odd.

''' 
def even_odd_sign(n):
    if n % 2 == 0:
        return 1
    else:
        return -1
"""
class selection_placement_net(nn.Module):
    def __init__(self, use_cuda, K, n_y, in_channel_unet = 3, out_channel = 1):
        super(selection_placement_net, self).__init__()
        self.use_cuda = use_cuda
        self.K = K
        self.n_y = n_y
        self.selection_net = selection_net(self.use_cuda, self.K)
        self.placement_net = placement_net(self.n_y, in_channel_unet, out_channel, self.use_cuda)
        if self.use_cuda == True:
            self.selection_net.cuda()
            self.placement_net.cuda()
        else:
            self.selection_net.cpu()
            self.placement_net.cpu()
    
    def forward(self, input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp):
        # Compute score values using the selection network
        Q_values_sel = self.selection_net(input1_selection_HM_6views, boxHM, input2_selection_ids)
        # Apply Gumbel-Softmax to the score values
        alpha = 900
        attention_weights = torch.softmax(alpha * Q_values_sel,dim =1)
        while torch.max(attention_weights).item() == 1:
                    alpha = alpha - 100
                    attention_weights = torch.softmax(alpha * Q_values_sel, dim =1)

        # Compute Q-values using the placement network
        Q_values_pla, selected_obj, orients = self.placement_net(input1_placement_rp_angles, input2_placement_HM_rp, boxHM, attention_weights)
        return Q_values_sel, Q_values_pla, selected_obj, orients, attention_weights
"""
class selection_net(nn.Module):
    def __init__(self, use_cuda, K): 
        super(selection_net, self).__init__()
        self.use_cuda = use_cuda
        self.K = K
        
        # Initialize network trunks with ResNet pre-trained on ImageNet
        self.backbones = nn.ModuleList([self.initialize_backbone() for _ in range(self.K)])
        
        # Add a spatial pooling layer to pool the (7, 7) features
        self.spatial_pooling = nn.AvgPool2d(kernel_size=(7, 7))
        
        self.final_selection_layers = final_conv_select_net(self.use_cuda, self.K)

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
        
        # Add an adaptive average pooling layer to ensure output is always 7x7
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((7, 7))
        )
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        backbone = backbone.to(device)

        return backbone

    def forward(self, input_1, input_2, item_ids, epsilon):
        if self.use_cuda:
                input_1 = input_1.cuda()
                input_2 = input_2.cuda()
                item_ids = item_ids.cuda()
    
        concatenated_features = []
        zero_masks = []
        for k in range(self.K):
            # Check if the input is all zeros
            is_all_zero = torch.all(input_1[:, k, :, :, :] == 0)
            zero_masks.append(is_all_zero)

            # Show the selected tensor
            # fig, axs = plt.subplots(1, 1, figsize=(10, 5))
            # axs.imshow(input_1[0, k, 0, :, :].detach().cpu().numpy(), cmap='viridis', origin='lower')
            # axs.set_xlabel('X')
            # axs.set_ylabel('Y')
            # plt.tight_layout()
            # plt.show()

            concatenated_input = torch.cat((input_1[:, k, :, :, :], input_2), dim=1)
            backbone_output = self.backbones[k](concatenated_input.float())
            pooled_features = self.spatial_pooling(backbone_output)
            pooled_features = pooled_features.squeeze(dim=-1).squeeze(dim=-1)
            concatenated_features.append(pooled_features)

        concatenated_features = torch.stack(concatenated_features, dim=1)
        Q_values = self.final_selection_layers(concatenated_features)

        # Convert zero_masks to a tensor and expand to match the shape of score_values
        zero_masks_tensor = torch.tensor(zero_masks, device=input_1.device).float().unsqueeze(0)
        zero_masks_tensor = zero_masks_tensor.expand_as(Q_values)

        # Ensure score_values for all-zero inputs remain zero
        Q_values = Q_values * (1 - zero_masks_tensor)

        
        # Apply Gumbel-Softmax to the score values
        alpha = 900
        attention_weights = torch.softmax(alpha * Q_values,dim =1)
        while torch.max(attention_weights).item() == 1:
            alpha = alpha - 100
            attention_weights = torch.softmax(alpha * Q_values, dim =1)
        
        """
        # EXPLOITATION EXPLORATION TRADE-OFF: EPSILON-GREEDY
        if np.random.rand() < epsilon:
            # Scegli un'azione casuale
            print(f'{red_light}Sto eseguendo EXPLORATION!{reset}') 
            selected_obj = np.random.choice(len(Q_values))  # Assumendo che q_values sia un array
        else:
            # Scegli l'azione con il massimo Q-value
            print(f'{red_light}Sto eseguendo EXPLOITATION{reset}')
            selected_obj = int(torch.argmax(Q_values, dim=1).detach().cpu().numpy())
        """  

        #PRENDE OGGETTO DA attention_weights
        # EXPLOITATION EXPLORATION TRADE-OFF: EPSILON-GREEDY
        if np.random.rand() < epsilon:
            # Scegli un'azione casuale
            print(f'{red_light}Sto eseguendo EXPLORATION!{reset}') 
            selected_obj = random.randint(0, self.K - 1)
        else:
            # Scegli l'azione con il massimo Q-value
            selected_obj = int(torch.argmax(Q_values).cpu().numpy())
            selected_obj = torch.argmax(attention_weights)
        
        selected_obj = int(torch.argmax(attention_weights).cpu().numpy())
        selected_obj_pybullet = int(item_ids.clone().cpu().detach()[selected_obj]) 
        
        Qvalue = Q_values[:, selected_obj]

        return Q_values, Qvalue, selected_obj_pybullet, attention_weights
    
    # Compute labels and backpropagate
    def backprop(self, Q_targets_tensor, Q_values_tensor, counter, counter_treshold):            

        loss = self.criterion(Q_values_tensor, Q_targets_tensor)
        loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
        print(f"{blue_light}\nComputing loss and gradients on Selection Network{reset}")
        print('Training loss: %f' % (loss))
        print('---------------------------------------') 
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Inspect gradients
        # print('SELECTION NETWORK GRAIDENTS:')
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
        #         print(f'Layer: {name} | Gradient mean: {param.grad.mean()} | Gradient std: {param.grad.std()}')
        #     else:
        #         print(f"Layer: {name} | No gradients computed")

        # Check for NaN gradients
        for name, param in self.named_parameters():           
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise ValueError("Gradient of {name} is NaN!")

        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         plt.figure(figsize=(10, 5))
        #         plt.title(f'Gradients for {name}')
        #         plt.hist(param.grad.cpu().numpy().flatten(), bins=50, log=True)
        #         plt.xlabel('Gradient Value')
        #         plt.ylabel('Count')
        #         plt.show()

        # if optimizer_step == True:
        if counter % counter_treshold == 0: 
        # if self.epoch % 4 == 0:             
            print(f"{blue_light}\nBackpropagating loss on Selection Network (manager network){reset}\n")
            print('---------------------------------------') 
            self.optimizer.step()
            print('---------------------------------------')           
            print(f"{purple}Selection Network trained on ", self.epoch+1, f"EPOCHS{reset}")    # ???????????????????????????????????????????????????????????????????????
            print('---------------------------------------')  
            self.optimizer.zero_grad()

        if self.use_cuda:
            torch.cuda.empty_cache()

        return loss
    
    
class final_conv_select_net(nn.Module):
    def __init__(self, use_cuda, K):
        super(final_conv_select_net, self).__init__()

        self.num_classes = K

        # Define fully connected layers for each branch
        self.fc_layers = nn.ModuleList([self.create_fc_layers() for _ in range(K)])
        
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        # Move model to GPU if use_cuda is True
        if use_cuda:
            self.cuda()
    
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
    def __init__(self, n_y, in_channel_unet = 3, out_channel = 1, use_cuda = True):
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

    def forward(self, roll_pitch_angles, input1, input2, attention_weights):
        if self.use_cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()

        batch_size = input1.size(0)
        self.n_rp = input1.size(2)
        K = input1.size(1)
        # Reshape input2 to match the batch size of input1
        input2 = input2.permute(0, 2, 3, 1).expand(batch_size, -1, -1, -1)
        # Reshape the attention weights to match the dimensions of the tensor
        weighted_input = input1 * attention_weights.view(1, K, 1, 1, 1, 1)
        selected_tensor = weighted_input.sum(dim=1, keepdim=True)

        # Show the selected tensor
        # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        # axs[0].imshow(selected_tensor[0,0,0,:,:,0].detach().cpu().numpy(), cmap='viridis', origin='lower')
        # axs[0].set_xlabel('X')
        # axs[0].set_ylabel('Y')
        # axs[1].imshow(input1[0,int(torch.argmax(attention_weights).cpu().numpy()),0,:,:,0].detach().cpu().numpy(), cmap='viridis', origin='lower')
        # axs[1].set_xlabel('X')
        # axs[1].set_ylabel('Y')
        # axs[2].imshow(input2[0,:,:,0].detach().cpu().numpy(), cmap='viridis', origin='lower')
        # axs[2].set_xlabel('X')
        # axs[2].set_ylabel('Y')
        # plt.tight_layout()
        # plt.show()

        ### selected_obj = int(torch.argmax(attention_weights).cpu().numpy())
        
        angles_yaw = torch.arange(0, 360, 360 / self.n_y).unsqueeze(0).to(input1.device)
        # Expand and reshape angles_yaw
        angles_yaw_expanded = angles_yaw.unsqueeze(0).expand(1,self.n_rp,self.n_y).reshape(-1, 1) # torch.Size([1, n_y])

        # Expand and reshape roll_pitch_angles_expanded
        roll_pitch_angles_expanded = roll_pitch_angles.unsqueeze(1).expand(-1, self.n_y, -1).reshape(-1, 2).to(input1.device) # torch.Size([n_rp, 2])

        # Concatenate along the last dimension
        orients = torch.cat((roll_pitch_angles_expanded, angles_yaw_expanded), dim=-1)  # torch.Size([n_rp*n_y, 3])
        
        input1_rp = selected_tensor.squeeze(1).unsqueeze(2).expand(-1, -1, self.n_y, -1, -1, -1) # torch.Size([1, n_rp, n_y, res, res, 2])
        
        unet_input = self.rotate_tensor_and_append_bbox(input1_rp, orients, input2 ) # torch.Size([n_rp*n_y, res, res, 3])
        Q_values = self.unet_forward(unet_input) # torch.Size([n_rp*n_y, res, res])
        
        
        orients = orients.cpu().detach().numpy()
        return Q_values, orients

    def unet_forward(self, x):
        ''' Reshape input '''
        x = x.permute(0, 3, 1, 2) # torch.Size([nrp*ny, 3, res, res])
        outputs = []
        for i in range(self.n_rp):
            xi = x[i*self.n_y:(i+1)*self.n_y] # torch.Size([n_y, 3, res, res])
            ''' Downsample'''
            xi = self.layer1_conv_block(xi)
            f1 = xi
            xi = self.layer2_Downsample(xi)
            xi = self.layer3_conv_block(xi)
            f2 = xi
            xi = self.layer4_Downsample(xi)
            xi = self.layer5_conv_block(xi)
            f3 = xi
            xi = self.layer6_Downsample(xi)
            xi = self.layer7_conv_block(xi)
            f4 = xi
            ''' Upsample'''
            xi = self.layer8_Downsample(xi)
            xi = self.layer9_conv_block(xi)
            xi = self.layer10_Upsample(xi, f4)
            xi = self.layer11_conv_block(xi)
            xi = self.layer12_Upsample(xi, f3)
            xi = self.layer13_conv_block(xi)
            xi = self.layer14_Upsample(xi, f2)
            xi = self.layer15_conv_block(xi)
            xi = self.layer16_Upsample(xi, f1)
            xi = self.layer17_conv_block(xi)

            xi = self.layer18(xi) # torch.Size([n_y, 1, res, res])
            outputs.append(torch.squeeze(xi)) # torch.Size([n_y, res, res])

        output = torch.cat(outputs, dim=0) # torch.Size([nrp*ny, res, res])
        return output

    def rotate_tensor_and_append_bbox(self, input1, orient, input2):
        # Generate rotation matrix
        angles = orient[:, 2] 
        rotation_matrices = []
        for i, angle in enumerate(angles):
            if self.use_cuda == True:
                angle = angle.cuda()
                rotation_matrix = torch.stack([
                    even_odd_sign(i) * torch.cos(angle), even_odd_sign(i) * -torch.sin(angle), torch.tensor(0.).cuda(),
                    even_odd_sign(i) * torch.sin(angle), even_odd_sign(i) * torch.cos(angle), torch.tensor(0.).cuda()
                ]).reshape(2, 3)
            else:
                rotation_matrix = torch.stack([
                    even_odd_sign(i) * torch.cos(angle), even_odd_sign(i) * -torch.sin(angle), torch.tensor(0.),
                    even_odd_sign(i) * torch.sin(angle), even_odd_sign(i) * torch.cos(angle), torch.tensor(0.)
                ]).reshape(2, 3)

            rotation_matrix = rotation_matrix.unsqueeze(0) #[1,2,3]
            rotation_matrices.append(rotation_matrix)

        rotation_matrix = torch.cat(rotation_matrices, dim=0) # [batch,2,3]
        # put channels before in the input 1 
        input1 = input1.permute(0,1,2,5,3,4).reshape(-1, *input1.shape[3:]) # [batch*nrp*ny, res, res, 2]
        input2 = input2.expand(input1.shape[0], *input2.shape[1:]) # [batch*nrp*ny, res, res, 1]

        if self.use_cuda == True :
            flow_grid_before = F.affine_grid(rotation_matrix.cuda(), input1.size(),align_corners=True)
            rotated_hm = F.grid_sample(input1.cuda(), flow_grid_before, mode='nearest', align_corners=True) #[batch*nrp*ny, res, res, 2]
            rotated_hm = rotated_hm.cuda()
            input2 = input2.cuda()
        elif self.use_cuda == False:
            flow_grid_before = F.affine_grid(rotation_matrix, input1.size(),align_corners=True)
            rotated_hm = F.grid_sample(input1, flow_grid_before, mode='nearest',align_corners=True) #[batch*nrp*ny, res, res, 2]

        # Concatenate the rotated heightmap with the box heightmap
        input_unet = torch.cat([rotated_hm,input2], dim=-1)

        return input_unet.float()
    
        # Compute labels and backpropagate
    def backprop(self, Q_targets_tensor, Q_values_tensor, replay_buffer_length, replay_batch_size, counter, counter_treshold):            

        loss = self.criterion(Q_values_tensor, Q_targets_tensor)
        loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
        print(f"{blue_light}\nComputing loss and gradients on Placement Network{reset}")
        print('Training loss: %f' % (loss))
        print('---------------------------------------') 
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.placement_net.parameters(), max_norm=1.0)

        # Inspect gradients
        # print('NETWORK GRAIDENTS:')
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
        #         print(f'Layer: {name} | Gradient mean: {param.grad.mean()} | Gradient std: {param.grad.std()}')
        #     else:
        #         print(f"Layer: {name} | No gradients computed")

        # Check for NaN gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise ValueError("Gradient of {name} is NaN!")

        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         plt.figure(figsize=(10, 5))
        #         plt.title(f'Gradients for {name}')
        #         plt.hist(param.grad.cpu().numpy().flatten(), bins=50, log=True)
        #         plt.xlabel('Gradient Value')
        #         plt.ylabel('Count')
        #         plt.show()

        # if optimizer_step == True:
        if replay_buffer_length >= replay_batch_size and counter % counter_treshold == 0:
            print(f"{blue_light}\nBackpropagating loss on Placement Network (worker network){reset}\n")
            self.optimizer.step()
            print(f"{purple}Placement Network trained on ", self.epoch+1, f"EPOCHS{reset}")
            print('---------------------------------------')  
            self.optimizer.zero_grad()
            self.epoch = self.epoch+1

        if self.use_cuda:
            torch.cuda.empty_cache()

        return loss
    
           
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
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    
if __name__ == '__main__':

    method = 'stage_2'
    batch_size = 1
    K = 1
    resolution = 50
    n_rp = 1
    cuda = False
    if cuda == True:
        input1_selection_HM_6views = torch.randn(batch_size, K, 6, resolution, resolution).cuda()  # object heightmaps at 6 views
        boxHM = torch.randn(batch_size, 1, resolution, resolution).cuda()                   # box heightmap
        input2_selection_ids = torch.randn(K).cuda()                                               # list of loaded ids
        input1_placement_rp_angles = torch.randn(n_rp,2).cuda()                                   # roll-pitch angles
        input2_placement_HM_rp = torch.randn(batch_size, K, n_rp, resolution, resolution, 2).cuda()    # object heightmaps at different roll-pitch angles
    else:
        input1_selection_HM_6views = torch.randn(batch_size, K, 6, resolution, resolution)
        boxHM = torch.randn(batch_size, 1, resolution, resolution)
        input2_selection_ids = torch.randn(K)
        input1_placement_rp_angles = torch.randn(n_rp,2)
        input2_placement_HM_rp = torch.randn(batch_size, K, n_rp, resolution, resolution, 2)
        
    model = selection_net(use_cuda = cuda, K = K, n_y = 4)
    model.train()


    #-- placement_net visulize
    # model_graph = draw_graph(model, input_data=[input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp],graph_name = 'model',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/')
    #model_graph.visual_graph
    
    Q_values, selected_obj, orients = model(input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp)
    indices_rpy, pixel_x, pixel_y =   0, int(resolution/2), int(resolution/2)
    Q_target = 0.1

    Q_target_tensor = torch.tensor(Q_target).float()
    Q_target_tensor = Q_target_tensor.expand_as(Q_values[indices_rpy, pixel_x, pixel_y])
    Q_target_tensor = Q_target_tensor.cuda() if model.use_cuda == True else Q_target_tensor
    criterion = torch.nn.SmoothL1Loss(reduction='mean') 
    loss = criterion(Q_values[ indices_rpy, pixel_x, pixel_y], Q_target_tensor)
    loss.backward()                                                     # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
    

    # Inspect gradients
    print(' NETWORK GRAIDENTS:')
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
        else:
            print(f"Layer: {name} | No gradients computed")

    def count_parameters_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() )

    placement_net = placement_net(n_y=4, in_channel_unet=3, out_channel=1, use_cuda=False)
    print('The number of trainable parameters in the placement net is: ' ,count_parameters_trainable(placement_net))
    print('The tot number of parameters in the placement net is: ' ,count_parameters(placement_net))
    print('--------------------------------------')
    selection_net = selection_net(use_cuda=False, K=K, method = method)
    print('The number of trainable parameters in the selection net is: ' ,count_parameters_trainable(selection_net))
    print('The tot number of parameters in the selection net is: ' ,count_parameters(selection_net))
    print('--------------------------------------')
    print('The number of trainable parameters in the selection-placement net is: ' , count_parameters_trainable(selection_net) + count_parameters_trainable(placement_net))
    print('The tot number of parameters in the selection-placement net is: ' , count_parameters(selection_net)+ count_parameters(placement_net))