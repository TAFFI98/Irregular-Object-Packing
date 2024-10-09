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
#from env import Env
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc
from experience_replay import ExperienceReplayBuffer
import random


class Trainer(object):
    def __init__(self, type_net, epsilon, epsilon_min, epsilon_decay, future_reward_discount=0.5, force_cpu = False, file_snapshot = None, load_snapshot = False ,K = 20, n_y = 4, epoch = 0, episode = 0):
        self.n_y = n_y           # number of discrete yaw orientations
        self.K = K               # total number of items to be packed
        self.epoch = epoch       # epoch counter
        self.episode = episode   # episode counter
        self.lr = 1e-4           # learning rate 1e-3 for stage 1 and 1e-4 for stage 2
        self.weight_decay = 2.5e-5 # weight decay

        self.epsilon = epsilon              # Valore iniziale per epsilon
        self.epsilon_min = epsilon_min      # Valore minimo per epsilon
        self.epsilon_decay = epsilon_decay  # Fattore di decrescita per epsilon
    
        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print(f"CUDA detected. Running with GPU acceleration.{reset}")
            self.use_cuda = True
        elif force_cpu:
            print(f"CUDA detected, but overriding with option '--cpu'. Running with only CPU.{reset}")
            self.use_cuda = False
        else:
            print(f"CUDA is *NOT* detected. Running with only CPU.{reset}")
            self.use_cuda = False

        if type_net == 'manager':
            # INITIALIZE NETWORK
            self.selection_net = selection_net(use_cuda = self.use_cuda, K = self.K)
            super
            
            if self.use_cuda == True:
                self.selection_net.cuda()
            else:
                self.selection_net.cpu()

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean') # Huber loss
            self.future_reward_discount = future_reward_discount

            # Load pre-trained model
            if load_snapshot:
                map_location = torch.device('cuda') if self.use_cuda else torch.device('cpu')
                self.selection_net.load_state_dict(torch.load(file_snapshot, map_location=map_location))
                print(f'{red}Pre-trained model snapshot loaded from: %s' % (file_snapshot),f'{reset}')
        
            # Convert model from CPU to GPU
            if self.use_cuda:
                self.selection_net = self.selection_net.cuda()
            
            # Set model to training mode
            self.selection_net.train()

            # Initialize optimizers
            self.optimizer_manager = torch.optim.Adam(self.selection_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.iteration = 0
            

        elif type_net == 'worker':
            # INITIALIZE NETWORK
            self.placement_net = placement_net(n_y = self.n_y, in_channel_unet = 3, out_channel = 1, use_cuda = self.use_cuda)
            if self.use_cuda == True:
                self.placement_net.cuda()
            else:
                self.placement_net.cpu()  
            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean') # Huber loss
            self.future_reward_discount = future_reward_discount

            # Load pre-trained model
            if load_snapshot:
                map_location = torch.device('cuda') if self.use_cuda else torch.device('cpu')
                elf.placement_net.load_state_dict(torch.load(file_snapshot, map_location=map_location))
                self.placement_net.load_state_dict(torch.load(file_snapshot))
                print(f'{red}Pre-trained model snapshot loaded from: %s' % (file_snapshot),f'{reset}')
        
            # Convert model from CPU to GPU
            if self.use_cuda:
                self.placement_net = self.placement_net.cuda()

            # Set model to training mode
            self.placement_net.train()
        
            # Initialize optimizers
            self.optimizer_worker = torch.optim.Adam(self.placement_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.iteration = 0
            
 
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
        Q_values, selected_obj, orients, attention_weights  = self.selection_placement_net.forward(input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp)
        selected_obj_pybullet = int(input2_selection_ids.clone().cpu().detach()[selected_obj]) 
        orients = orients.cpu().detach().numpy()
        return  Q_values , selected_obj_pybullet, orients, attention_weights
    
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

    """
    # Compute labels and backpropagate
    def backprop(self, Q_targets_tensor, Q_values_tensor, replay_buffer_length, replay_batch_size, counter):            

        loss = self.criterion(Q_values_tensor, Q_targets_tensor)
        loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
        print(f"{blue_light}\nComputing loss and gradients on network{reset}")
        print('Training loss: %f' % (loss))
        print('---------------------------------------') 
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.selection_placement_net.parameters(), max_norm=1.0)

        # Inspect gradients
        # print('NETWORK GRAIDENTS:')
        # for name, param in self.selection_placement_net.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
        #         print(f'Layer: {name} | Gradient mean: {param.grad.mean()} | Gradient std: {param.grad.std()}')
        #     else:
        #         print(f"Layer: {name} | No gradients computed")

        # Check for NaN gradients
        for name, param in self.selection_placement_net.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise ValueError("Gradient of {name} is NaN!")

        # for name, param in self.selection_placement_net.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         plt.figure(figsize=(10, 5))
        #         plt.title(f'Gradients for {name}')
        #         plt.hist(param.grad.cpu().numpy().flatten(), bins=50, log=True)
        #         plt.xlabel('Gradient Value')
        #         plt.ylabel('Count')
        #         plt.show()

        # if optimizer_step == True:
        if replay_buffer_length >= replay_batch_size and counter % 15 ==0:
            print(f"{blue_light}\nBackpropagating loss on worker network{reset}\n")
            self.optimizer_worker.step()
            print(f"{purple}Network trained on ", self.epoch+1, f"EPOCHS{reset}")
            print('---------------------------------------')  
            self.optimizer_worker.zero_grad()
            self.epoch = self.epoch+1

            if self.epoch % 4 == 0:
                print(f"{blue_light}\nBackpropagating loss on manager network{reset}\n")
                print('---------------------------------------') 
                self.optimizer_manager.step()
                print('---------------------------------------')           
                print(f"{purple}Network trained on ", int(self.epoch/ 4)+1, f"EPOCHS{reset}")
                print('---------------------------------------')  
                self.optimizer_manager.zero_grad()

        if self.use_cuda:
            torch.cuda.empty_cache()

        return loss
    """

    # Compute labels and backpropagate
    def backprop_sel(self, Q_targets_tensor, Q_values_tensor, epoch_pla, epoch_treshold):            

        loss = self.criterion(Q_values_tensor, Q_targets_tensor)
        loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
        print(f"{blue_light}\nComputing loss and gradients on Selection Network{reset}")
        print('Training loss: %f' % (loss))
        print('---------------------------------------') 
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.selection_net.parameters(), max_norm=1.0)

        # Inspect gradients
        print('SELECTION NETWORK GRAIDENTS:')
        for name, param in self.selection_net.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
                print(f'Layer: {name} | Gradient mean: {param.grad.mean()} | Gradient std: {param.grad.std()}')
            else:
                print(f"Layer: {name} | No gradients computed")

        # Check for NaN gradients
        for name, param in self.selection_net.named_parameters():           
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

        #if epoch_pla > 0 and epoch_pla % epoch_treshold == 0:             
        print(f"{blue_light}\nBackpropagating loss on Selection Network (manager network){reset}\n")
        print('---------------------------------------') 
        self.optimizer_manager.step()
        print('---------------------------------------')           
        print(f"{purple}Selection Network trained on ", self.epoch+1, f"EPOCHS{reset}")    # ???????????????????????????????????????????????????????????????????????
        print('---------------------------------------')  
        self.optimizer_manager.zero_grad()
        self.epoch = self.epoch+1

        if self.use_cuda:
            torch.cuda.empty_cache()

        return loss

    # Compute labels and backpropagate
    def backprop_pla(self, Q_targets_tensor, Q_values_tensor, replay_buffer_length, replay_batch_size, counter, counter_treshold):            

        loss = self.criterion(Q_values_tensor, Q_targets_tensor)
        loss.backward() # loss.backward() computes the gradient of the loss with respect to all tensors with requires_grad=True. 
        print(f"{blue_light}\nComputing loss and gradients on Placement Network{reset}")
        print('Training loss: %f' % (loss))
        print('---------------------------------------') 
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.placement_net.parameters(), max_norm=1.0)

        # Inspect gradients
        print('PLACEMENT NETWORK GRAIDENTS:')
        for name, param in self.placement_net.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name} | Gradients computed: {param.grad.size()}")
                print(f'Layer: {name} | Gradient mean: {param.grad.mean()} | Gradient std: {param.grad.std()}')
            else:
                print(f"Layer: {name} | No gradients computed")

        # Check for NaN gradients
        for name, param in self.placement_net.named_parameters():
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

        #if replay_buffer_length >= replay_batch_size and counter % counter_treshold == 0:
        print(f"{blue_light}\nBackpropagating loss on Placement Network (worker network){reset}\n")
        self.optimizer_worker.step()
        print(f"{purple}Placement Network trained on ", self.epoch+1, f"EPOCHS{reset}")
        print('---------------------------------------')  
        self.optimizer_worker.zero_grad()
        self.epoch = self.epoch+1

        if self.use_cuda:
            torch.cuda.empty_cache()

        return loss

    # NEW VODE THAT PLOT FROM STAR OF THE TRAINIBG, NOT JUST THE EPOCHS IN THE CURRENT SESSION
    def save_and_plot_loss(self, epoch_number, loss, folder, max_images=4):
        '''
        epoch_number: current epoch number
        loss: current loss value
        folder: folder to save the plots
        max_images: maximum number of images to save
        This function updates the loss values, saves them, and plots them
        '''
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # File paths for epoch and loss data
        data_file = os.path.join(folder, 'loss_data.pkl')

        # Load existing data if available
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                list_epochs_for_plot, losses = pickle.load(f)
        else:
            list_epochs_for_plot = []
            losses = []

        # Append new values to the lists
        list_epochs_for_plot.append(epoch_number)
        losses.append(loss)

        # Save the updated lists to a file
        with open(data_file, 'wb') as f:
            pickle.dump((list_epochs_for_plot, losses), f)

        # Plot the data
        plt.figure()
        plt.plot(list_epochs_for_plot, losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Save the plot to a file
        plot_file = os.path.join(folder, 'loss_plot.png')
        plt.savefig(plot_file)
        plt.close()

        # Get a list of all files in the directory
        files = glob.glob(os.path.join(folder, '*'))

        # If there are more than max_images files
        if len(files) > max_images:
            # Sort the files by modification time
            files.sort(key=os.path.getmtime)

            # Remove the oldest files
            for file in files[:len(files) - max_images]:
                os.remove(file)

    def save_and_plot_reward(self, epoch_number, reward, folder, max_images=4):
        '''
        epoch_number: current epoch number
        reward: current reward value
        folder: folder to save the plots
        max_images: maximum number of images to save
        This function updates the reward values, saves them, and plots them
        '''
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # File paths for epoch and reward data
        data_file = os.path.join(folder, 'reward_data.pkl')

        # Load existing data if available
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                list_epochs_for_plot, rewards = pickle.load(f)
        else:
            list_epochs_for_plot = []
            rewards = []

        # Append new values to the lists
        list_epochs_for_plot.append(epoch_number)
        rewards.append(reward)

        # Save the updated lists to a file
        with open(data_file, 'wb') as f:
            pickle.dump((list_epochs_for_plot, rewards), f)

        # Plot the data
        plt.figure()
        plt.plot(list_epochs_for_plot, rewards, label='Reward')
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()

        # Save the plot to a file
        plot_file = os.path.join(folder, 'reward_plot.png')
        plt.savefig(plot_file)
        plt.close()

        # Get a list of all files in the directory
        files = glob.glob(os.path.join(folder, '*'))

        # If there are more than max_images files
        if len(files) > max_images:
            # Sort the files by modification time
            files.sort(key=os.path.getmtime)

            # Remove the oldest files
            for file in files[:len(files) - max_images]:
                os.remove(file)

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

    
    def ebstract_max(self, Q_values):
      Q_max = torch.max(Q_values)  # Trova il massimo valore nel tensore Q_values
      return float(Q_max.cpu()) 


    # NUOVO CODICE CON AGGIUNTA METODO EPSILON-GREEDY
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

        attempt = 1

        # Flatten the Q_values tensor
        Q_values_flat = Q_values.view(-1)
        Q_size = Q_values.size()

        if torch.rand(1).item() < self.epsilon:
          # Exploration: select a random index
          print(f'{red_light}Sto eseguendo EXPLORATION!{reset}')          
          random_index = torch.randint(0, Q_values_flat.size(0), (1,)).item()
        else:
          # Exploitation: select the index with the maximum Q value
          print(f'{red_light}Sto eseguendo EXPLOITATION!{reset}')
          _, random_index = torch.max(Q_values_flat, 0)

        # Unravel the index to get the 3D coordinates
        index_tensor = torch.tensor(random_index)
        index = torch.unravel_index(index_tensor, Q_size)
        indices_rpy = int(index[0])
        pixel_x = int(index[1])
        pixel_y = int(index[2])

        r = float(orients[indices_rpy, 0])
        p = float(orients[indices_rpy, 1])
        y = float(orients[indices_rpy, 2])

        # Pack chosen item with predicted pose
        target_euler = [r, p, y]
        # Compute z coordinate
        _, Hb_selected_obj, obj_length, obj_width, offsets = env.item_hm(chosen_item_index, target_euler)
        offset_pointminz_COM = offsets[4]

        z_lowest_point = env.get_z(BoxHeightMap, Hb_selected_obj, pixel_x, pixel_y, obj_length, obj_width)
        del(Hb_selected_obj, BoxHeightMap)
        gc.collect()
        z = z_lowest_point + offset_pointminz_COM
        target_pos = [pixel_x * env.box_size[0] / env.resolution, pixel_y * env.box_size[1] / env.resolution, z]  # m
        transform = np.empty(6,)
        transform[0:3] = target_euler
        transform[3:6] = target_pos
        print('----------------------------------------')
        print(f'{yellow}Check packing validity for chosen item with index', chosen_item_index, 'with candidate pose n', 1, ': \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')

        # Pack item
        BoxHeightMap, stability_of_packing, old_pos, old_quater, collision, limits_obj_line_ids, height_exceeded_before_pack = env.pack_item_check_collision(chosen_item_index, transform, offsets)

        # Use the collision result
        if collision:
            print(f'{red_light}Collision detected!{reset}')
            # Place the object back to its original position since this predicted pose would not be valid
            env.removeAABB(limits_obj_line_ids)
            BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
            packed = False
            #Q_max = Q_values[indices_rpy, pixel_x, pixel_y]
        elif height_exceeded_before_pack:
            print(f'{red_light}Box height exceeded before packing action!{reset}')
            # Place the object back to its original position since this predicted pose would not be valid
            env.removeAABB(limits_obj_line_ids)
            BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
            packed = False
            #Q_max = Q_values[indices_rpy, pixel_x, pixel_y]
        else:
            print(f'{green_light}No collision detected.{reset}')
            # Check if the height of the box is exceeded
            height_exceeded = env.check_height_exceeded(box_heightmap=BoxHeightMap, box_height=env.box_size[2])
            if height_exceeded:
                print(f'{red_light}Box height exceeded after packing action!{reset}')
                # Place the object back to its original position since this predicted pose would not be valid
                env.removeAABB(limits_obj_line_ids)
                BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                packed = False
                #Q_max = Q_values[indices_rpy, pixel_x, pixel_y]
            else:
                print(f'{green_light}Box height not exceeded.{reset}')
                print('--------------------------------------')
                print(f'{green}Packed item with id', chosen_item_index, f'successfully!{reset}')
                print(f'{green}with pose: \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')
                print('--------------------------------------')
                print(f'{purple_light}>Stability is:', stability_of_packing, f'{reset}')
                env.removeAABB(limits_obj_line_ids)
                packed = True
                #Q_max = Q_values[indices_rpy, pixel_x, pixel_y]

        return indices_rpy, pixel_x, pixel_y, BoxHeightMap, stability_of_packing, packed, attempt


    def check_placement_validity_TENTATIVES(self, env, Q_values, orients, BoxHeightMap, chosen_item_index, max_attempts):
        '''
        env: environment object
        Q_values: predicted Q_values #[batch, n_y, n_rp, res, res]
        orients: numpy array of shape (n_rp*n_y, 3) - roll pitch yaw angles
        BoxHeightMap: heightmap of the box
        chosen_item_index: index of the object to be packed
        epsilon: probability for exploration (selecting random action)
        max_attempts: max number of attempts to try packing the object

        output:
        indices_rpy: index of the selected orientation
        pixel_x: x coordinate of the selected pixel
        pixel_y: y coordinate of the selected pixel
        BoxHeightMap: updated heightmap of the box
        stability_of_packing: stability of the packed object
        packed: boolean indicating if the object was packed
        Q_max: maximum Q_value
        '''
        
        # Flatten the Q_values tensor for easier sorting
        Q_values_flat = Q_values.view(-1)
        Q_size = Q_values.size()
        sorted_values, sorted_indices = torch.sort(Q_values_flat, descending=True)
        _, _, box_height = env.box_size

        # Epsilon-greedy strategy: Explore (choose random index) or exploit (choose best Q_value)
        if random.random() < self.epsilon:
            # Exploration: choose a random index
            exploration = True
            print(f'{bold}Sto eseguendo EXPLORATION!{reset}')
        else:
            # Exploitation: choose the best Q_value
            print(f'{bold}Sto eseguendo EXPLOITATION!{reset}')
            exploration = False

        # Attempt to pack the object with a maximum number of attempts
        for attempt in range(max_attempts):
            
            # Epsilon-greedy strategy: Explore (choose random index) or exploit (choose best Q_value)
            if exploration:
                # Exploration: choose a random index
                # chosen_index = random.randint(0, len(Q_values_flat) - 1)
                chosen_index = torch.randint(0, Q_values_flat.size(0), (1,)).item()
                print(f"Exploring: Random action chosen at attempt {attempt + 1}")
            else:
                # Exploitation: choose the best Q_value
                chosen_index = sorted_indices[attempt]
                print(f"Exploiting: Best action chosen at attempt {attempt + 1}")

            index_tensor = torch.tensor(chosen_index)
            index = torch.unravel_index(index_tensor, Q_size)
            indices_rpy = int(index[0])
            pixel_x = int(index[1])
            pixel_y = int(index[2])

            r = float(orients[indices_rpy, 0])
            p = float(orients[indices_rpy, 1])
            y = float(orients[indices_rpy, 2])

            # Pack the chosen item with the predicted pose
            target_euler = [r, p, y]

            # Compute z coordinate
            _, Hb_selected_obj, obj_length, obj_width, offsets = env.item_hm(chosen_item_index, target_euler)
            offset_pointminz_COM = offsets[4]

            # Uncomment to visualize the heightmap of the predicted pose of the object and the difference between the heightmap of the box and the object
            #env.visualize_object_heightmaps(Hb_selected_obj, None, target_euler, only_top = True)
            #env.visualize_object_heightmaps_3d(BoxHeightMap-Hb_selected_obj, None, target_euler, only_top = True)

            z_lowest_point = env.get_z(BoxHeightMap, Hb_selected_obj, pixel_x, pixel_y, obj_length, obj_width)
            del(Hb_selected_obj, BoxHeightMap)
            gc.collect()
            z = z_lowest_point + offset_pointminz_COM
            target_pos = [pixel_x * env.box_size[0] / env.resolution, pixel_y * env.box_size[1] / env.resolution, z]  # m

            transform = np.empty(6,)
            transform[0:3] = target_euler
            transform[3:6] = target_pos

            print('----------------------------------------')
            print(f'{yellow}Check packing validity for chosen item with index {chosen_item_index}, with candidate pose attempt {attempt + 1}:{reset}')
            print(f'{yellow}> orientation (r, p, y): {target_euler}{reset}')
            print(f'{yellow}> pixel coordinates: [{pixel_x}, {pixel_y}]{reset}')
            print(f'{yellow}> position (m): {target_pos}{reset}')
            # print(f'{yellow}Check packing validity for chosen item with index', chosen_item_index, 'with candidate pose n ',attempt+1,': \n> orientation (r,p,y):', target_euler, '\n> pixel coordinates: [', pixel_x, ',', pixel_y, '] \n> position (m):', target_pos, f'{reset}')
            
            # Pack item
            BoxHeightMap, stability_of_packing, old_pos, old_quater, collision, limits_obj_line_ids, height_exceeded_before_pack = env.pack_item_check_collision(chosen_item_index, transform, offsets)

            if collision:
                print(f'{red_light}Collision detected!{reset}')
                # Place the object back to its original position since this predicted pose would not be valid
                env.removeAABB(limits_obj_line_ids)
                BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                continue

            if height_exceeded_before_pack:
                print(f'{red_light}Box height exceeded before packing! {reset}')
                # Place the object back to its original position since this predicted pose would not be valid
                env.removeAABB(limits_obj_line_ids)
                BoxHeightMap = env.unpack_item(chosen_item_index, old_pos, old_quater)
                continue

            elif not collision and not height_exceeded_before_pack:
                print(f'{green_light}No collision detected.{reset}')
                # Check if the height of the box is exceeded
                height_exceeded = env.check_height_exceeded(box_heightmap=BoxHeightMap, box_height=box_height)
                if height_exceeded:
                    print(f'{red_light}Box height exceeded after packing! {reset}')
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
                    Q_max = Q_values[indices_rpy, pixel_x, pixel_y]
                    return indices_rpy, pixel_x, pixel_y, BoxHeightMap, stability_of_packing, packed, attempt+1

        # If not packed after all attempts, return failure
        packed = False
        Q_max = Q_values[indices_rpy, pixel_x, pixel_y]
        # return indices_rpy, pixel_x, pixel_y, False, 0, packed, attempt+1
        return indices_rpy, pixel_x, pixel_y, BoxHeightMap, 0, packed, attempt+1

        

    def update_epsilon_linear(self):
        '''
        Aggiorna il valore di epsilon per l'epsilon decay.
        '''
        eps = self.epsilon - self.epsilon_decay
        self.epsilon = max(eps, self.epsilon_min)
        print(f'{blue_light}Nuovo valore di epsilon: {self.epsilon}{reset}')

    def update_epsilon_exponential(self):
        '''
        Aggiorna il valore di epsilon per l'epsilon decay.
        '''
        eps = self.epsilon * self.epsilon_decay
        self.epsilon = max(eps, self.epsilon_min)
        print(f'{blue_light}Nuovo valore di epsilon: {self.epsilon}{reset}')

    def update_epsilon_inverse(self):
        '''
        Aggiorna il valore di epsilon per l'epsilon decay.
        '''
        t = self.episode + 1 #                                          NON SICURA CHE SIA CORRETTO IL CONETGGIO 
        eps = self.epsilon_min / ( 1 + self.epsilon_decay ** t)
        self.epsilon = max(eps, self.epsilon_min)
        print(f'{blue_light}Nuovo valore di epsilon: {self.epsilon}{reset}')    


    def save_snapshot(self, type_net, snapshot_dir, max_snapshots=5):
        """
        Save snapshots of the trained models.
        """
        # Create the directory if it doesn't exist
        #snapshot_dir = os.path.join('snapshots', 'models', folder_name)
        os.makedirs(snapshot_dir, exist_ok=True)

        snapshot_file = os.path.join(snapshot_dir, f'network_episode_{self.episode}_epoch_{self.epoch}.pth')
        
        if type_net == 'manager':
          torch.save(self.selection_net.state_dict(), snapshot_file)
        else:
          torch.save(self.placement_net.state_dict(), snapshot_file)

        # Get a list of all files in the directory
        files = glob.glob(os.path.join(snapshot_dir, '*'))

        # If there are more than 5 files
        if len(files) > max_snapshots:
                # Sort the files by modification time
                files.sort(key=os.path.getmtime)

                # Remove the oldest file
                os.remove(files[0])

        return snapshot_file