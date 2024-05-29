# -*- coding: utf-8 -*-
"""
Created on  Apr 3 2024

@author: Taffi
"""
from fonts_terminal import *
import os
import time
import glob
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import  placement_net, selection_net
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc

class Trainer(object):
    def __init__(self, method= 'stage_1', future_reward_discount=0.5, force_cpu = False, file_snapshot_worker = None, file_snapshot_manager = None, load_snapshot_worker = False, load_snapshot_manager = False ,K = 20, n_y = 16, epoch = 0, episode = 0):
        
        self.n_y = n_y # number of discrete yaw orientations
        self.method = method # stage_1 or stage_2
        self.K = K # total number of items to be packed
        self.epoch = epoch # epoch counter
        self.episode = episode # episode counter
        lr = 1e-4 # learning rate
        momentum = 0.9 # momentum
        weight_decay = 2e-5 # weight decay

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print(f"{bold}CUDA detected. Running with GPU acceleration.{reset}")
            self.use_cuda = True
        elif force_cpu:
            print(f"{bold}CUDA detected, but overriding with option '--cpu'. Running with only CPU.{reset}")
            self.use_cuda = False
        else:
            print(f"{bold}CUDA is *NOT* detected. Running with only CPU.{reset}")
            self.use_cuda = False

        # IN BOTH STAGES  TRAIN THE PLACEMENT NETWORK
        if self.method == 'stage_1' or self.method == 'stage_2':
            # INITIALIZE WORKER NETWORK
            self.worker_network = placement_net(n_y = self.n_y, in_channel_unet = 3, out_channel = 1, use_cuda = self.use_cuda)
                
            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean') # Huber loss
            self.future_reward_discount = future_reward_discount
            if self.use_cuda:
                    self.criterion = self.criterion.cuda()

            # Load pre-trained model
            if load_snapshot_worker:
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
                self.manager_network.load_state_dict(torch.load(file_snapshot_manager))
                print('Pre-trained model snapshot loaded from: %s' % (file_snapshot_manager))
            self.manager_network.train()
            self.optimizer_manager = torch.optim.SGD(self.manager_network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            # Convert model from CPU to GPU
            if self.use_cuda:
                self.manager_network = self.manager_network.cuda()

        
        print('---------------------------------------------------')
        print(f"{bold}TRAINER INITIALIZED.{reset}")
        print(f"{bold}METHOD: %s" % (self.method),f"{reset}")
        print('Manager network snapshot: ' , file_snapshot_manager)
        print('Worker network snapshot: ' , file_snapshot_worker)
        print('---------------------------------------------------')

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
    def forward_worker_network(self, env, roll_pitch_angles, input_placement_network_1, input_placement_network_2):
        '''
        input_placement_network_1: numpy array of shape (batch, n_rp, res, res, 2) 
        input_placement_network_2: numpy array of shape (batch, res, res, 1) - bounding box heightmap
        env: environment object
        output:
        Q_values: predicted Q_values
        '''
        #-- placement network
        Q_values  = self.worker_network.forward(env, roll_pitch_angles, torch.tensor(input_placement_network_1, requires_grad=False),torch.tensor(input_placement_network_2, requires_grad=False))

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
        return current_reward, yt

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
            if self.method == 'stage_2':
                self.optimizer_manager.zero_grad()

            self.optimizer_worker.zero_grad()

            loss_value = 0

            if self.use_cuda:
                    loss = self.criterion(Q_values.float().cuda(), label_Q.float().cuda())
            else:
                    loss = self.criterion(Q_values.float(), label_Q.float())
            
            loss = loss.sum()
            loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
            loss_value = loss.cpu().data.numpy()
            loss_value = loss_value/2

            print(f'{blue_light}------------------ Training loss: %f' % (loss_value),f'------------------{reset}')
            if self.epoch % 4 == 0 and self.method == 'stage_2':
                print(f'{blue_light}--> Backpropagating loss on manager network{reset}')
                self.optimizer_manager.step()

            print(f'{blue_light}-->Backpropagating loss on worker network{reset}')
            self.optimizer_worker.step()

            self.epoch = self.epoch+1
            if self.use_cuda:
                torch.cuda.empty_cache()
            return loss_value

    def save_and_plot_loss(self, list_epochs_for_plot, losses, folder, max_images = 4):
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Get the number of epochs
        num_epochs = int(list_epochs_for_plot[-1])

        # Save the lists to a file
        with open(os.path.join(folder, f'loss_epoch_{num_epochs}.pkl'), 'wb') as f:
            pickle.dump((list_epochs_for_plot, losses), f)

        # Plot the data
        plt.figure()
        plt.plot(list_epochs_for_plot, losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(os.path.join(folder, f'loss_epoch_{num_epochs}.png'))
        plt.close()
        # Get a list of all files in the directory
        files = glob.glob(os.path.join(folder, '*'))

        # If there are more than 5 files
        if len(files) > max_images:
                # Sort the files by modification time
                files.sort(key=os.path.getmtime)

                # Remove the oldest file
                os.remove(files[0])
                os.remove(files[1])

    def save_and_plot_reward(self, list_epochs_for_plot, rewards, folder, max_images = 4):
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Get the number of epochs
        num_epochs = int(list_epochs_for_plot[-1])

        # Save the lists to a file
        with open(os.path.join(folder, f'reward_epoch_{num_epochs}.pkl'), 'wb') as f:
            pickle.dump((list_epochs_for_plot, rewards), f)

        # Plot the data
        plt.figure()
        plt.plot(list_epochs_for_plot, rewards)
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(os.path.join(folder, f'reward_epoch_{num_epochs}.png'))
        plt.close()
        # Get a list of all files in the directory
        files = glob.glob(os.path.join(folder, '*'))

        # If there are more than 5 files
        if len(files) > max_images:
                # Sort the files by modification time
                files.sort(key=os.path.getmtime)

                # Remove the oldest file
                os.remove(files[0])
                os.remove(files[1])

    # Visualize the predictions of the worker network: Q_values
    def visualize_Q_values(self, Q_values, show = True):
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
        if show == True:
            cv2.namedWindow("Q Values", cv2.WINDOW_NORMAL)
            cv2.imshow("Q Values", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return canvas
        
    # Check if the predicted pose is allowed: Collision with box margins and exceeded height of the box
    def check_placement_validity(self, env, Q_values, roll_pitch_angles, BoxHeightMap, chosen_item_index):
        '''
        env: environment object
        Q_values: predicted Q_values #[batch, n_y, n_rp, res, res]
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
        Q_size = Q_values.size()
        sorted_values, sorted_indices = torch.sort(Q_values_flat, descending=True)
        _, _, box_height = env.box_size

        del(Q_values,Q_values_flat,sorted_values)
        gc.collect()
        # Determine the max number of tentatives to pack the object for each batch
        tentatives = 20
        tentatives_sets= 0
        # Iterate over the batches
        # Calculate the start and end indices for this batch
        start_index = tentatives_sets * tentatives
        end_index = min((tentatives_sets + 1) * tentatives, len(sorted_indices))

        # Get the batch of indices
        batch_indices = sorted_indices[start_index:end_index]
        del(sorted_indices)
        gc.collect()
        # Unravel the indices in this batch
        for k,i in enumerate(batch_indices):
            index = torch.unravel_index(i, Q_size)
            # Extract indices along each dimension
            indices_y = int(index[1])
            indices_rp = int(index[2])
            pixel_x = int(index[3])
            pixel_y = int(index[4])
            del(index)
            gc.collect()
            r = float(roll_pitch_angles[indices_rp,0])
            p = float(roll_pitch_angles[indices_rp,1])
            y = float(indices_y * (360 / self.n_y))

            # Pack chosen item with predicted pose
            target_euler = [r,p,y]
            # Compute z coordinate
            _,Hb_selected_obj, obj_length, obj_width, offsets = env.item_hm(chosen_item_index, target_euler)
            offset_pointminz_COM = offsets[4]
            # Uncomment to visualize the heightmap of the predicted pose of the object and the difference between the heightmap of the box and the object
            #env.visualize_object_heightmaps(Hb_selected_obj, None, target_euler, only_top = True)
            #env.visualize_object_heightmaps_3d(BoxHeightMap-Hb_selected_obj, None, target_euler, only_top = True)
            z_lowest_point = env.get_z(BoxHeightMap, Hb_selected_obj, pixel_x, pixel_y, obj_length, obj_width)
            del(Hb_selected_obj, BoxHeightMap)
            gc.collect()
            z = z_lowest_point + offset_pointminz_COM
            target_pos = [pixel_x * env.box_size[0]/env.resolution,pixel_y * env.box_size[1]/env.resolution, z] # m
            transform = np.empty(6,)
            transform[0:3] = target_euler
            transform[3:6] = target_pos
            print('----------------------------------------')
            print(f'{yellow}Check packing validity for chosen item with index', chosen_item_index, 'with candidate pose n ',k+1,': \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')
            
            # Pack item
            BoxHeightMap, stability_of_packing, old_pos, old_quater, collision, limits_obj_line_ids, height_exceeded_before_pack = env.pack_item_check_collision(chosen_item_index , transform, offsets)

            # Use the collision result
            if collision:
                print(f'{red_light}Collision detected!{reset}')
                # Place the object back to its original position since this predicted pose would not be valid
                env.removeAABB(limits_obj_line_ids)
                BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                continue
            if height_exceeded_before_pack:
                print(f'{red_light}Box height exceeded before packing action!{reset}')
                # Place the object back to its original position since this predicted pose would not be valid
                env.removeAABB(limits_obj_line_ids)
                BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                continue
            elif not collision and not height_exceeded_before_pack:
                print(f'{green_light}No collision detected.{reset}')
                # Check if the height of the box is exceeded
                height_exceeded = env.check_height_exceeded(box_heightmap = BoxHeightMap, box_height = box_height)
                if height_exceeded:
                    print(f'{red_light}Box height exceeded after packing action!{reset}')
                    # place the object back to its original position since this predicted pose would not be valid
                    env.removeAABB(limits_obj_line_ids)
                    BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                    continue
                elif not height_exceeded:
                    print(f'{green_light}Box height not exceeded.{reset}')
                    print('--------------------------------------')
                    print(f'{green}Packed chosen item!! {reset}')
                    print(f'{green}with pose: \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')                   
                    print('--------------------------------------')
                    print(f'{purple_light}>Stability is:', stability_of_packing,f'{reset}')
                    env.removeAABB(limits_obj_line_ids)
                    packed = True
                    return indices_rp, indices_y, pixel_x, pixel_y, BoxHeightMap, stability_of_packing, packed
            
        packed = False
        return False, False, False, False, False, False, packed

        
    
    def save_snapshot(self, max_snapshots=5):
        """
        Save snapshots of the trained models.
        """
        os.makedirs('snapshots/models/', exist_ok=True)
        torch.save(self.worker_network.state_dict(), f'snapshots/models/worker_network_episode_{self.episode}_epoch_{self.epoch}.pth')

        # If there's a manager network, save it as well
        if self.method == 'stage_2':
            torch.save(self.manager_network.state_dict(), f'snapshots/models/manager_network_episode_{self.episode}_epoch_{self.epoch}.pth')


        # Get a list of all files in the directory
        files = glob.glob(os.path.join('snapshots/models/', '*'))

        # If there are more than 5 files
        if len(files) > max_snapshots:
                # Sort the files by modification time
                files.sort(key=os.path.getmtime)

                # Remove the oldest file
                os.remove(files[0])

        return f'snapshots/models/manager_network_episode_{self.episode}_epoch_{self.epoch}.pth', f'snapshots/models/worker_network_episode_{self.episode}_epoch_{self.epoch}.pth'

