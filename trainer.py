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
from models import  selection_placement_net
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc

class Trainer(object):
    def __init__(self, method= 'stage_1', future_reward_discount=0.5, force_cpu = False, file_snapshot = None, load_snapshot = False ,K = 20, n_y = 4, epoch = 0, episode = 0):
        
        self.n_y = n_y # number of discrete yaw orientations
        self.method = method # stage_1 or stage_2
        self.K = K # total number of items to be packed
        self.epoch = epoch # epoch counter
        self.episode = episode # episode counter
        self.lr = 1e-4 # learning rate
        self.momentum = 0.9 # momentum
        self.weight_decay = 2e-5 # weight decay

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

        # INITIALIZE  NETWORK
        self.selection_placement_net = selection_placement_net(K = self.K, n_y = self.n_y, use_cuda = self.use_cuda, method = self.method)
            
        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean') # Huber loss
        self.future_reward_discount = future_reward_discount

        # Load pre-trained model
        if load_snapshot:
            self.selection_placement_net.load_state_dict(torch.load(file_snapshot))
            print(f'{red}Pre-trained model snapshot loaded from: %s' % (file_snapshot),f'{reset}')

        # Convert model from CPU to GPU
        if self.use_cuda:
                self.selection_placement_net = self.selection_placement_net.cuda()

        # Set model to training mode
        self.selection_placement_net.train()
        
        # Initialize optimizers
        self.optimizer_manager = torch.optim.SGD(self.selection_placement_net.selection_net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.optimizer_worker = torch.optim.SGD(self.selection_placement_net.placement_net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.iteration = 0
        
        print('---------------------------------------------------')
        print(f"{bold}TRAINER INITIALIZED.{reset}")
        print(f"{bold}METHOD: %s" % (self.method),f"{reset}")
        print('---------------------------------------------------')
 
    def forward_network(self, input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp):
        '''
        input_placement_network_1: numpy array of shape (batch, n_rp, res, res, 2) 
        input_placement_network_2: numpy array of shape (batch, res, res, 1) - bounding box heightmap
        env: environment object
        output:
        Q_values: predicted Q_values
        '''
        #-- placement network
        Q_values, selected_obj, orients  = self.selection_placement_net.forward(input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp)
        selected_obj_pybullet = int(input2_selection_ids.clone().cpu().detach()[selected_obj]) 
        orients = orients.cpu().detach().numpy()
        return  Q_values , selected_obj_pybullet, orients
    
    # Compute target Q_target
    def get_Qtarget_value(self, Q_max, prev_obj, current_obj, env):
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
        Q_target: target Q_value
        '''
        # Compute current reward
        current_reward = env.Reward_function(prev_obj, current_obj)

        # Compute expected reward:
        future_reward = Q_max

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        Q_target = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f ' % (Q_target))
        return current_reward, Q_target

    # Compute labels and backpropagate
    def backprop(self, Q_values, Q_target, indices_rpy, pixel_x, pixel_y, optimizer_step):
            '''
            Q_max: predicted Q_values
            Q_target: target Q_value
            indices_rp: roll and pitch index of the selected pixel
            indices_y: yaw index of the selected pixel
            pixel_x: x coordinate of the selected pixel
            pixel_y: y coordinate of the selected pixel
            label_weight: weight of the label in the Huber loss
            This function computes the labels and backpropagates the loss across the networks
            '''
            

            if self.use_cuda:
                Q_target_tensor = torch.tensor(Q_target).cuda().float()
                Q_target_tensor = Q_target_tensor.expand_as(Q_values[indices_rpy, pixel_x, pixel_y])
            else:
                Q_target_tensor = torch.tensor(Q_target).float()
                Q_target_tensor = Q_target_tensor.expand_as(Q_values[ indices_rpy, pixel_x, pixel_y])
            
            loss = self.criterion(Q_values[indices_rpy, pixel_x, pixel_y], Q_target_tensor)
            loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
            print(f'{blue_light}------------------ Training loss: %f' % (loss),f'------------------{reset}')
            
            #Inspect gradients
            print('WORKER NETWORK GRAIDENTS:')
            for name, param in self.selection_placement_net.named_parameters():
                if param.grad is not None:
                    print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
                else:
                    print(f"Layer: {name} | No gradients computed")
 
            if optimizer_step == True:
                print(f'{blue_light}-->Backpropagating loss on worker network {reset}')
                self.optimizer_worker.step()
                self.optimizer_worker.zero_grad()
                self.epoch = self.epoch+1

                if self.epoch % 4 == 0 and self.method == 'stage_2':
                    print(f'{blue_light}--> Backpropagating loss on manager network{reset}')
                    self.optimizer_manager.step()
                    self.optimizer_manager.zero_grad()

            if self.use_cuda:
                torch.cuda.empty_cache()

            return loss
    
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
        num_rotations = Q_values.shape[0]
        resolution = Q_values.shape[1]
        grid_rows = int(num_rotations/4)
        grid_cols = int(num_rotations/4)
        border_size = 10  # Size of the border in pixels

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

        return canvas
        
    # Check if the predicted pose is allowed: Collision with box margins and exceeded height of the box
    def check_placement_validity(self, env, Q_values, orients, BoxHeightMap, chosen_item_index):
        '''
        env: environment object
        Q_values: predicted Q_values #[batch, n_y, n_rp, res, res]
        roll_pitch_angles: numpy array of shape (n_rp*n_y, 2) - roll pitch yaw angles
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

        # Determine the max number of tentatives to pack the object for each batch
        tentatives = 10
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
                    print(f'{green}Packed chosen item!! {reset}')
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

        
    
    def save_snapshot(self, max_snapshots=5):
        """
        Save snapshots of the trained models.
        """
        os.makedirs('snapshots/models/', exist_ok=True)
        torch.save(self.selection_placement_net.state_dict(), f'snapshots/models/network_episode_{self.episode}_epoch_{self.epoch}.pth')

        # Get a list of all files in the directory
        files = glob.glob(os.path.join('snapshots/models/', '*'))

        # If there are more than 5 files
        if len(files) > max_snapshots:
                # Sort the files by modification time
                files.sort(key=os.path.getmtime)

                # Remove the oldest file
                os.remove(files[0])

        return  f'snapshots/models/network_episode_{self.episode}_epoch_{self.epoch}.pth'

