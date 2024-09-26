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
from models import  placement_net
from models import  selection_net
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc

class Tester(object):
    def __init__(self, method= 'stage_1', future_reward_discount=0.5, force_cpu = False, file_snapshot = None, load_snapshot = False ,K = 20, n_y = 4, epoch = 0, episode = 0):
        
        self.n_y = n_y           # number of discrete yaw orientations
        self.method = method     # stage_1 or stage_2
        self.K = K               # total number of items to be packed
        self.epoch = epoch       # epoch counter
        self.episode = episode   # episode counter

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

        # INITIALIZE NETWORK
        self.selection_placement_net = selection_placement_net(K = self.K, n_y = self.n_y, use_cuda = self.use_cuda, method = self.method)
        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean') # Huber loss
        self.future_reward_discount = future_reward_discount
        
        # Load pre-trained model
        if load_snapshot:
            if torch.cuda.is_available() and not force_cpu:
                self.selection_placement_net.load_state_dict(torch.load(file_snapshot))
            else:
                # Load the state dict from the file with the map_location argument
                state_dict = torch.load(file_snapshot, map_location=torch.device('cpu'))
                # Load the state dict into the model
                self.selection_placement_net.load_state_dict(state_dict)
                print("Model state dict loaded successfully on CPU")
            print(f'{red}Pre-trained model snapshot loaded from: %s' % (file_snapshot),f'{reset}')

        # Convert model from CPU to GPU
        if self.use_cuda:
                self.selection_placement_net = self.selection_placement_net.cuda()

        # Set model to training mode
        self.selection_placement_net.eval()
        
        print('---------------------------------------------------')
        print(f"{bold}TESTER INITIALIZED.{reset}")
        print(f"{bold}METHOD: %s" % (self.method),f"{reset}")
        print('---------------------------------------------------')
 
    def forward_network(self, input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp):
        '''
        input1_selection_HM_6views: input heightmaps of the 6 views of the objects
        boxHM: heightmap of the box
        input2_selection_ids: indices of the objects
        input1_placement_rp_angles: roll and pitch angles of the objects
        input2_placement_HM_rp: heightmap of the objects at different roll and pitch angles
        output:
        Q_values: predicted Q_values
        selected_obj: selected object
        orients: array with the roll, pitch and yaw angles considered ordered
        '''
        #-- placement network
        with torch.no_grad():
            Q_values, selected_obj, orients  = self.selection_placement_net.forward(input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp)
            selected_obj_pybullet = int(input2_selection_ids.clone().cpu().detach()[selected_obj]) 
            orients = orients.cpu().detach().numpy()
        return  Q_values , selected_obj_pybullet, orients
    
    # Compute target Q_target
    def get_Qtarget_value(self, Q_max, prev_obj, current_obj, env):
        '''
        prev_obj: previous objective function value
        current_obj: current objective function value
        env: environment object
        output:
        Q_target: target Q_value according to Bellman equation
        '''
        # Compute current reward 
        print(f"{blue_light}\nComputing Reward fuction {reset}\n")
        current_reward = env.Reward_function(prev_obj, current_obj)

        # Compute expected reward:
        future_reward = Q_max

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        Q_target = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f ' % (Q_target))
        print('---------------------------------------')           
        return current_reward, Q_target

    # Compute labels and backpropagate
    def loss(self, Q_values, Q_target, indices_rpy, pixel_x, pixel_y):
            '''
            This function computes the gradients and backpropagates the loss across the networks

            Q_values: predicted Q_values
            Q_target: target Q_value
            indices_rpy: index of the selected orientation
            pixel_x: x coordinate of the selected pixel
            pixel_y: y coordinate of the selected pixel
            output:
            loss: loss value
            '''
            

            if self.use_cuda:
                Q_target_tensor = torch.tensor(Q_target).cuda().float()
                Q_target_tensor = Q_target_tensor.expand_as(Q_values[indices_rpy, pixel_x, pixel_y])
            else:
                Q_target_tensor = torch.tensor(Q_target).float()
                Q_target_tensor = Q_target_tensor.expand_as(Q_values[ indices_rpy, pixel_x, pixel_y])
            
            loss = self.criterion(Q_values[indices_rpy, pixel_x, pixel_y], Q_target_tensor)
            print(f"{blue_light}\nComputing loss and gradients on network{reset}")
            print('Training loss: %f' % (loss))
            print('---------------------------------------') 
            return loss
    
    def save_and_plot_loss(self, list_epochs_for_plot, losses, folder, max_images = 4):
        '''
        list_epochs_for_plot: list of epochs
        losses: list of losses
        folder: folder to save the plots
        max_images: maximum number of images to save
        This function saves the loss values and plots them
        '''
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
        '''
        list_epochs_for_plot: list of epochs
        rewards: list of rewards
        folder: folder to save the plots
        max_images: maximum number of images to save
        This function saves the reward values and plots them
        '''
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
    def visualize_Q_values(self, Q_values, show = True, save = False, path = 'snapshots/Q_values/'):
        '''
        Q_values: numpy array of shape (1, n_rp, n_y, resolution, resolution)
        output: visualization of Q_values using colormaps
        This function visualizes the Q_values using colormaps
        '''
        Q_values = Q_values.cpu().detach().numpy()
        num_rotations = Q_values.shape[0]
        resolution = Q_values.shape[1]
        grid_rows = int(num_rotations/2)
        grid_cols = int(num_rotations/2)
        border_size = 10 # Size of the border in pixels

        # Adjust the size of the canvas to account for the borders
        canvas = np.zeros((grid_rows * (resolution + 2*border_size), 
                        grid_cols * (resolution + 2*border_size), 3), dtype=np.uint8)

        for i in range(grid_rows): 
            for j in range(grid_cols):  
                Q_values_vis = Q_values[j, :, :].copy()
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
        if save == True:
            os.makedirs('snapshots/Q_values/', exist_ok=True)
            cv2.imwrite(path + f'Q_values_episode_{self.episode}_epoch_{self.epoch}.png', canvas)

        return canvas
        
    # Check if the predicted pose is allowed: Collision with box margins and exceeded height of the box
    def check_placement_validity(self, env, Q_values, orients, BoxHeightMap, chosen_item_index):
        '''
        env: environment object
        Q_values: predicted Q_values #[batch, n_y, n_rp, res, res]
        orients: numpy array of shape (n_rp*n_y, 3) - roll pitch yaw angles
        BoxHeightMap: heightmap of the box
        chosen_item_index: index of the object to be packed

        output:
        indices_rpy: index of the selected orientation
        pixel_x: x coordinate of the selected pixel
        pixel_y: y coordinate of the selected pixel
        BoxHeightMap: updated heightmap of the box
        stability_of_packing: stability of the packed object
        packed: boolean indicating if the object was packed
        Q_max: maximum Q_value

        This function tries to pack the chosen item with the predicted pose and checks if the placement is valid 
        (no collision with box margins and box height not exceeded).
        '''
        # Flatten the Q_values tensor and sort it in descending order
        Q_values_flat = Q_values.view(-1)
        Q_size = Q_values.size()
        sorted_values, sorted_indices = torch.sort(Q_values_flat, descending=True)
        _, _, box_height = env.box_size

        # Determine the max number of tentatives to pack the object for each batch
        tentatives = 1
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
            indices_rpy = int(index[0])
            pixel_x = int(index[1])
            pixel_y = int(index[2])

            r = float(orients[indices_rpy,0])
            p = float(orients[indices_rpy,1])
            y = float(orients[indices_rpy,2])

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
                    print(f'{green}Packed item with id ', chosen_item_index, f'successfully!{reset}')
                    print(f'{green}with pose: \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')                   
                    print('--------------------------------------')
                    print(f'{purple_light}>Stability is:', stability_of_packing,f'{reset}')
                    env.removeAABB(limits_obj_line_ids)
                    packed = True
                    Q_max = Q_values[indices_rpy,pixel_x,pixel_y]
                    return indices_rpy, pixel_x, pixel_y, BoxHeightMap, stability_of_packing, packed,  float(Q_max.cpu())
            
        packed = False
        Q_max = Q_values[indices_rpy,pixel_x,pixel_y]
        return indices_rpy, pixel_x, pixel_y, False, 0 , packed, float(Q_max.cpu())


