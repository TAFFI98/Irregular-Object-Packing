# -*- coding: utf-8 -*-
"""
Created on  Apr 3 2024

@author: Taffi
"""

import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import  placement_net, selection_net
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, method= 'stage_2', future_reward_discount=0.5,force_cpu ='False', file_snapshot_worker = 'snapshots/worker_network_1.pth', file_snapshot_manager='snapshots/manager_network_1.pth', load_snapshot_worker = False, load_snapshot_manager = False ,K=20, n_y=16):
        
        self.n_y = n_y # number of discrete yaw orientations
        self.method = method # stage_1 or stage_2
        self.K = K # total number of items to be packed
        self.epoch = 0 # epoch counter
        lr = 1e-4 # learning rate
        momentum = 0.9 # momentum
        weight_decay = 2e-5 # weight decay

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # IN BOTH STAGES  TRAIN THE PLACEMENT NETWORK
        if self.method == 'stage_1' or self.method == 'stage_2':
            # INITIALIZE WORKER NETWORK
            self.worker_network = placement_net(n_y = self.n_y, in_channel_unet=3, out_channel=1, use_cuda = self.use_cuda)
                
            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            self.future_reward_discount = future_reward_discount
            if self.use_cuda:
                    self.criterion = self.criterion.cuda()

            # Load pre-trained model
            if load_snapshot_worker:
                self.epoch = int(file_snapshot_worker.split('_')[-1].split('.')[0])
                self.worker_network.load_state_dict(torch.load(file_snapshot_worker))
                print('Pre-trained model snapshot loaded from: %s' % (file_snapshot_worker))

            # Convert model from CPU to GPU
            if self.use_cuda:
                    self.worker_network = self.worker_network.cuda()

            # Set model to training mode
            self.worker_network.train()
            # Initialize optimizer
            self.optimizer_worker = torch.optim.SGD(self.worker_network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            self.iteration = 0

        # STAGE 2: TRAIN ALSO SELECTION NETWORK 
        if self.method == 'stage_2':

            # INITIALIZE SELECTION NETWORK
            self.manager_network = selection_net(use_cuda = self.use_cuda, K = K)
            # Load pre-trained model
            if load_snapshot_manager:   
                self.epoch = int(file_snapshot_manager.split('_')[-1].split('.')[0])
                self.manager_network.load_state_dict(torch.load(file_snapshot_manager))
                print('Pre-trained model snapshot loaded from: %s' % (file_snapshot_manager))
            self.manager_network.train()
            self.optimizer_manager = torch.optim.SGD(self.manager_network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            # Convert model from CPU to GPU
            if self.use_cuda:
                self.manager_network = self.manager_network.cuda()

    # Compute forward pass through manager network to select the object to be packed
    def forward_manager_network(self, input1_selection_network, input2_selection_network, env):
        '''
        input1_selection_network: numpy array of shape (batch, K, 6, resolution, resolution)
        input2_selection_network: numpy array of shape (batch, 1, resolution, resolution)
        env: environment object
        output: 
        chosen_item_index: index of the object to be packed
        score_values: scores of the objects to be packed
        '''
        # Pass input data through model
        chosen_item_index, score_values = self.manager_network.forward(torch.tensor(input1_selection_network, requires_grad=False),torch.tensor(input2_selection_network, requires_grad=False),env.loaded_ids)
        print('----------------------------------------')
        print('Computed Manager network predictions. The object to be packed has been chosen.')

        return chosen_item_index, score_values

    # Compute forward pass through worker network to select the pose of the object to be packed
    def forward_worker_network(self, input_placement_network_1, input_placement_network_2, roll_pitch_angles):
        '''
        input_placement_network_1: numpy array of shape (batch, n_rp, res, res, 2) 
        input_placement_network_2: numpy array of shape (batch, res, res, 1) - bounding box heightmap
        roll_pitch_angles: numpy array of shape (n_rp, 2) - roll and pitch angles 
        output:
        Q_values: predicted Q_values
        '''
        #-- placement network
        Q_values  = self.worker_network.forward(torch.tensor(input_placement_network_1, requires_grad=False),torch.tensor(input_placement_network_2, requires_grad=False),roll_pitch_angles)
        print('----------------------------------------')
        print('Computed Worker network predictions. The pose of the object to be packed has been chosen.')

        return  Q_values 
    
    # Compute target Q_value yt
    def get_Qtarget_value(self, Q_values, indices_rp, indices_y, pixel_x, pixel_y, prev_obj, current_obj, env):
        '''
        Q_values: predicted Q_values
        indices_rp: roll and pitch index of the selected pixel
        indices_y: yaw index of the selected pixel
        pixel_x: x coordinate of the selected pixel
        pixel_y: y coordinate of the selected pixel
        prev_obj: previous objective function value
        current_obj: current objective function value
        env: environment object
        output:
        yt: target Q_value
        '''
        # Compute current reward
        current_reward = env.Reward_function(prev_obj, current_obj)

        # Compute expected reward:
        future_reward = float(Q_values[0,indices_rp,indices_y,pixel_x,pixel_y])

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        yt = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f ' % (yt))
        return yt

    # Compute labels and backpropagate
    def backprop(self, Q_values, yt, indices_rp, indices_y, pixel_x, pixel_y, label_weight=1):
            '''
            Q_values: predicted Q_values
            yt: target Q_value
            indices_rp: roll and pitch index of the selected pixel
            indices_y: yaw index of the selected pixel
            pixel_x: x coordinate of the selected pixel
            pixel_y: y coordinate of the selected pixel
            label_weight: weight of the label in the Huber loss
            This function computes the labels and backpropagates the loss across the networks
            '''
            
            # Compute labels to be able to pass gradients only through the selected action (pixel and orientation channel)
            label_Q = torch.zeros_like((Q_values))
            label_Q[:,indices_rp,indices_y,pixel_x,pixel_y] = yt * label_weight 
            # Compute loss and backward pass
            self.optimizer_worker.zero_grad()
            self.optimizer_manager.zero_grad()
            loss_value = 0
            if self.use_cuda:
                    loss = self.criterion(Q_values.float().cuda(), label_Q.float().cuda(), requires_grad=False)
            else:
                    loss = self.criterion(Q_values.float(), label_Q.float())
            
            loss = loss.sum()
            loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
            loss_value = loss.cpu().data.numpy()
            loss_value = loss_value/2

            print('----------------------------------------------------------------Training loss: %f' % (loss_value),'-------------------------------------------------------------------')
            if self.epoch % 4 == 0:
                print('!!!!!!!!Backpropagating on manager network!!!!!!!!')
                self.optimizer_manager.step()
            print('!!!!!!!!Backpropagating on worker network!!!!!!!!')
            self.optimizer_worker.step()
            self.epoch= self.epoch+1

    # Visualize the predictions of the worker network: Q_values
    def visualize_Q_values(self, Q_values):
        '''
        Q_values: numpy array of shape (1, n_rp, n_y, resolution, resolution)
        output: visualization of Q_values using colormaps
        This function visualizes the Q_values using colormaps
        '''
        Q_values = Q_values.cpu().detach().numpy()
        num_rotations = Q_values.shape[1]
        num_yaw = Q_values.shape[2]
        resolution = Q_values.shape[3]
        grid_rows = int(num_rotations/4)
        grid_cols = int(num_yaw/4)
        border_size = 10  # Size of the border in pixels

        # Adjust the size of the canvas to account for the borders
        canvas = np.zeros((grid_rows * (resolution + 2*border_size), 
                        grid_cols * (resolution + 2*border_size), 3), dtype=np.uint8)

        for i in range(grid_rows): 
            for j in range(grid_cols):  
                Q_values_vis = Q_values[0, i*4, j, :, :].copy()
                Q_values_vis = np.clip(Q_values_vis, 0, 1)
                Q_values_vis = cv2.applyColorMap((Q_values_vis*255).astype(np.uint8), cv2.COLORMAP_JET)

                # Add a border around Q_values_vis
                Q_values_vis = cv2.copyMakeBorder(Q_values_vis, border_size, border_size, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # Adjust the placement of Q_values_vis on the canvas to account for the borders
                canvas[i*(resolution+2*border_size):i*(resolution+2*border_size)+(resolution+2*border_size), 
                        j*(resolution+2*border_size):j*(resolution+2*border_size)+(resolution+2*border_size), :] = Q_values_vis

        return canvas
        
    # Check if the predicted pose is allowed: Collision with box margins and exceeded height of the box
    def check_placement_validity(self, env, Q_values, roll_pitch_angles, BoxHeightMap, chosen_item_index):
        '''
        env: environment object
        Q_values: predicted Q_values
        roll_pitch_angles: numpy array of shape (n_rp, 2) - roll and pitch angles
        BoxHeightMap: heightmap of the box
        chosen_item_index: index of the object to be packed

        output:
        indices_rp: roll and pitch index of the selected pixel
        indices_y: yaw index of the selected pixel
        pixel_x: x coordinate of the selected pixel
        pixel_y: y coordinate of the selected pixel
        BoxHeightMap: updated heightmap of the box
        stability_of_packing: boolean indicating if the placement is stable

        This function tries to pack the chosen item with the predicted pose and checks if the placement is valid 
        (no collision with box margins and box height not exceeded).
        '''
        # Flatten the Q_values tensor and sort it in descending order
        Q_values_flat = Q_values.view(-1)
        sorted_values, sorted_indices = torch.sort(Q_values_flat, descending=True)
        _, _, box_height = env.box_size
        # Iterate over the sorted indices
        for i in sorted_indices:
            # Unravel the index
            index = torch.unravel_index(i, Q_values.size())

            # Extract indices along each dimension
            indices_y = int(index[1])
            indices_rp = int(index[2])
            pixel_x = int(index[3])
            pixel_y = int(index[4])

            r = float(roll_pitch_angles[indices_rp,0])
            p = float(roll_pitch_angles[indices_rp,1])
            y = float(indices_y * (360 / self.n_y))


            print('----------------------------------------')
            print('Packing chosen item with predicted pose...')
            # Pack chosen item with predicted pose
            target_euler = [r,p,y]
            # Compute z coordinate
            _,Hb_selected_obj,obj_length,obj_width = env.item_hm(chosen_item_index, target_euler )
            z = env.get_z(BoxHeightMap,Hb_selected_obj,pixel_x, pixel_y,obj_length,obj_width)
            target_pos = [pixel_x * env.box_size[0]/env.resolution,pixel_y * env.box_size[1]/env.resolution, z] # m
            transform = np.empty(6,)
            transform[0:3] = target_euler
            transform[3:6] = target_pos
            BoxHeightMap, stability_of_packing, old_pos, old_quater = env.pack_item(chosen_item_index , transform)
            # Check collision with box margins 
            BBOX_object = env.compute_object_bbox(chosen_item_index)
            # Show the bounding box of the object to check collisions viusllay
            bbox_obj_line_ids = env.drawAABB(BBOX_object, width=1)
            collision = self.bounding_box_collision(env.bbox_box, BBOX_object)

            # Use the collision result
            if collision:
                print("Collision detected!")
                # Place the object back to its original position since this predicted pose would not be valid
                env.unpack_item(chosen_item_index, old_pos, old_quater)
                # Remove the bounding box of the object
                # Restore the position of the box since the invalid packing could have modified it
                env.removeAABB(bbox_obj_line_ids)
                continue
            else:
                print("No collision.")
                # Check if the height of the box is exceeded
                height_exceeded = self.check_height_exceeded(box_heightmap = BoxHeightMap, box_height = box_height)
                if height_exceeded:
                    print("Box height exceeded!")
                    # place the object back to its original position since this predicted pose would not be valid
                    env.unpack_item(chosen_item_index, old_pos, old_quater)
                    # Restore the position of the box since the invalid packing could have modified it
                    env.removeAABB(bbox_obj_line_ids)
                    continue
                else:
                    print("Box height not exceeded.")
                    print('--------------------------------------')
                    print('Packed chosen item ')
                    print('Is the placement stable?', stability_of_packing)
                    print('Loaded ids after packing: ', env.loaded_ids)
                    print('Packed ids after packing: ', env.packed)
                    print('UnPacked ids after packing: ', env.unpacked)
                    print('--------------------------------------')
                    env.removeAABB(bbox_obj_line_ids)
                    return indices_rp, indices_y, pixel_x, pixel_y, BoxHeightMap, stability_of_packing
        
    def bounding_box_collision(self, box1, box2):
        '''
        box1: bounding box of the box
        box2: bounding box of the object
        
        output: boolean indicating if the bounding box of the object is entirely inside the bounding box of the box (sensibility: mm)
        '''
        # Each box is a tuple of (min_x, min_y, min_z, max_x, max_y, max_z)
        min_x1, min_y1, _, max_x1, max_y1, _ = [round(x, 3) for x in box1]
        min_x2, min_y2, max_x2, max_y2 = [round(x, 3) for x in [box2[0,0],box2[0,1],box2[1,0],box2[1,1]]]

        # Check if box2 is entirely inside box1
        return not (min_x1 <= min_x2 and max_x1 >= max_x2 and
                min_y1 <= min_y2 and max_y1 >= max_y2 )

    def check_height_exceeded(self,box_heightmap, box_height):
        '''
        box_heightmap: heightmap of the box
        box_height: height of the box
        output: boolean indicating if the height of the box is exceeded
        '''
        # Compute the new heightmap after placing the object
        max_z = box_heightmap.max()

        # Check if any height in the new heightmap exceeds the box height
        return max_z > box_height
    
    def save_snapshot(self):
        """
        Save snapshots of the trained models.
        """
        torch.save(self.worker_network.state_dict(), f'snapshots/worker_network_{self.epoch}.pth')

        # If there's a manager network, save it as well
        if self.method == 'stage_2':
            torch.save(self.manager_network.state_dict(), f'snapshots/manager_network_{self.epoch}.pth')

