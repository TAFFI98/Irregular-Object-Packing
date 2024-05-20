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

    def forward(self, input1, input2):
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
                unet_input = self.rotate_tensor_and_append_bbox(input1_rp, angle, input2, batch_size, i) #[batch,3,res,res]
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
    
    def rotate_tensor_and_append_bbox(self, input1, angle, input2, batch_size,i ):
 
        # Generate rotation matrix
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
            
        #Uncomment to Show the rotated heightmaps
        fig, axs = plt.subplots(1, 5, figsize=(22, 4))
        axs[0].imshow(input1[0, 0, :, :].detach().cpu().numpy())
        axs[0].set_title(f"Original Top Heightmap")
        axs[0].axis('off')
        axs[1].imshow(input1[0, 1, :, :].detach().cpu().numpy())
        axs[1].set_title(f"Original Bottom Heightmap")
        axs[1].axis('off')           
        axs[2].imshow(rotated_hm[0, 0, :, :].detach().cpu().numpy())
        axs[2].set_title(f"Rotated Top Heightmap (angle={angle}°)")
        axs[2].axis('off')
        axs[3].imshow(rotated_hm[0, 1, :, :].detach().cpu().numpy())
        axs[3].set_title(f"Rotated Bottom Heightmap (angle={angle}°)")
        axs[3].axis('off')
        axs[4].imshow(input2[0, 0, :, :].detach().cpu().numpy())
        axs[4].set_title(f"Box Heightmap")
        axs[4].axis('off')
        plt.show()


        
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

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    
if __name__ == '__main__':
        

    #-- Path with the URDF files
    obj_folder_path = 'objects/'
    
    #-- PyBullet Environment setup 
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 50)

    #-- Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()

    #-- Draw Box
    env.draw_box(width=5)

    #-- Load items 
    item_numbers = np.arange(2,4)
    item_ids = env.load_items(item_numbers)

    for i in range(500):
        p.stepSimulation()

    #-- Compute Box HeightMap: shape (resolution, resolution)
    BoxHeightMap = env.box_heightmap()
    #env.visualize_box_heightmap_3d()
    #env.visualize_box_heightmap()
    print(' The box heightmap is: ', BoxHeightMap)

    for i in range(500):
        p.stepSimulation()

    #-- Compute the 6 views heightmaps for alle the loaded objects and store them in a numpy array

    principal_views = {
        "front": [0, 0, 0],
        "back": [180, 0, 0],
        "left": [0, -90, 0],
        "right": [0, 90, 0],
        "top": [-90, 0, 0],
        "bottom": [90, 0, 0]
    }
    six_views_all_items = []
    for id_ in item_ids:
        six_views_one_item = []
        for view in principal_views.values():
            Ht,_,obj_length,obj_width = env.item_hm(id_, view )
            env.visualize_object_heightmaps(id_, view, only_top = False )
            env.visualize_object_heightmaps_3d(id_, view, only_top = False )
            six_views_one_item.append(Ht) 
        six_views_one_item = np.array(six_views_one_item)
        six_views_all_items.append(six_views_one_item)
    six_views_all_items = np.array(six_views_all_items) # (n_loaded_items, 6, resolution, resolution)

    # -- Visualize selection network - I choose the first item
    six_views_chosen_item = six_views_all_items[0,:,:,:] # (6, resolution, resolution)
    
    # -- I concatenate the box heightmap    
    input_selection_network = np.concatenate((np.expand_dims(BoxHeightMap, axis=0), six_views_chosen_item), axis=0) # (7, resolution, resolution)
   
    # to account for batch = 1
    input_selection_network = np.expand_dims(input_selection_network, axis=0) 
    
    #-- feat_extraction_select_net visulization 
    feat_extraction_select_net = draw_graph(feat_extraction_select_net(use_cuda=False), input_data=[torch.tensor(input_selection_network.astype('float32'))],graph_name = 'feat_extraction_select_net',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/')
    feat_extraction_select_net.visual_graph

    #-- final_conv_select_net visulization 
    batch_size = 30
    K = 10
    final_conv_select_net_graph = draw_graph(final_conv_select_net(use_cuda=False,K = K), input_size=(batch_size,512,K),graph_name = 'final_conv_select_net',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/')
    final_conv_select_net_graph.visual_graph


    #-- Choose one Item and compute HeightMaps varying first roll, pitch and then yaw (according to the paper)
    item_id = item_ids[0]
    
    #-- Discretize roll, pitch and yaw
    roll_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    pitch_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    num_roll, num_pitch = roll_angles.shape[0], pitch_angles.shape[0]
    yaw_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    num_yaws = yaw_angles.shape[0]

    item_heightmaps_RP = []
    roll_pitch_angles = []

    for r,roll in enumerate(roll_angles):
            for p,pitch in enumerate(pitch_angles):
                orient = [roll, pitch, 0]
                roll_pitch_angles.append(np.array([roll,pitch]))
                Ht,Hb = env.item_hm(item_id, orient )
                # env.visualize_object_heightmaps(item_id, orient)
                # env.visualize_object_heightmaps_3d(item_id,orient )
                Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                item_heightmap = np.concatenate((Ht, Hb), axis=2)
                item_heightmaps_RP.append(item_heightmap)  

    item_heightmaps_RP = np.asarray(item_heightmaps_RP)  

    num_rp = item_heightmaps_RP.shape[0]
    item_heightmaps_RPY = np.empty(shape=(num_yaws, num_rp, item_heightmaps_RP.shape[1],item_heightmaps_RP.shape[2], 2)) # shape: (num_yaws, num_RP, resolution, resolution, 2)

    for j,yaw in enumerate(yaw_angles):
        for i in range(num_rp):
                    # Rotate Ht
                    Ht = item_heightmaps_RP[i,:,:,0]
                    rotated_Ht = rotate(Ht, yaw, reshape=False)
                    item_heightmaps_RPY[j,i,:,:,0] = rotated_Ht

                    # Rotate Hb
                    Hb = item_heightmaps_RP[i,:,:,1]
                    rotated_Hb = rotate(Hb, yaw, reshape=False)
                    item_heightmaps_RPY[j,i,:,:,1] = rotated_Hb

    # Selection of j = 1 (determines yaw) and i = 3 (determines roll and pitch) to test the networks  - from item_heightmaps_RPY of shape: (num_yaws, num_RP, resolution, resolution, 2) -
    
    input_placement_network = np.transpose(np.concatenate((np.expand_dims(BoxHeightMap, axis=2), item_heightmaps_RPY[1,3,:,:]), axis=2), (2, 0, 1))
   
    # to accounto for batch = 1
    input_placement_network = np.expand_dims(input_placement_network, axis=0) 
    
    #-- placement_net visulize
    placement_net_graph = draw_graph(placement_net(), input_data=[torch.tensor(input_placement_network.astype('float32'))],graph_name = 'placement_net',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/')
    placement_net_graph.visual_graph

    #- Visualize BoxHeightMap, Ht, Bb for the chosen orientation
    yaw = yaw_angles[1]
    roll = roll_pitch_angles[3][0]
    pitch = roll_pitch_angles[3][1]
    orient = [roll,pitch,yaw]
    env.visualize_object_heightmaps(item_id, orient)
    env.visualize_object_heightmaps_3d(item_id,orient )


    #-- conv_block visualize
    batch_size = 1
    conv_block_graph = draw_graph(conv_block(in_c=3,out_c=1), input_size=(batch_size,3,200,200),graph_name = 'conv_block',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/', graph_dir ='TB')
    conv_block_graph.visual_graph

    #-- Downsample visualize
    Downsample_graph = draw_graph(Downsample(channel=1), input_size=(batch_size,1,32,32),graph_name = 'Downsample',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/', graph_dir ='LR')
    Downsample_graph.visual_graph

    #-- Upsample visualize
    Upsample_graph = draw_graph(Upsample(channel=3), input_data=[torch.rand(batch_size,3,32,32), torch.rand(batch_size,3,64,64)],graph_name = 'Upsample',save_graph= True, directory= 'Irregular-Object-Packing/models_plot/',  graph_dir ='LR')
    Upsample_graph.visual_graph
