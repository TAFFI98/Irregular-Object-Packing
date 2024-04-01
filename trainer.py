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
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, method= 'stage_2', future_reward_discount=0.5, is_testing='False',force_cpu ='False',load_snapshot=False, snapshot_file='', use_cuda=False,K=20):

        self.method = method
        self.K = K

        lr = 1e-4
        momentum = 0.9 
        weight_decay = 2e-5

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

        # STAGE 2: TRAIN BOTH PLACEMENT AND SELECTION NETWORKS TOGETHER with Q_learning
        if self.method == 'stage_2':

            # Initialize worker and manager networks
            self.manager_network = selection_net(use_cuda = self.use_cuda, K = K)
            self.worker_network = placement_net(use_cuda = self.use_cuda)
            
            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            self.future_reward_discount = future_reward_discount
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

            # Load pre-trained model
            if load_snapshot:
                self.manager_network.load_state_dict(torch.load(snapshot_file))
                self.worker_network.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

            # Convert model from CPU to GPU
            if self.use_cuda:
                self.manager_network = self.manager_network.cuda()
                self.worker_network = self.worker_network.cuda()

            # Set models to training mode
            self.manager_network.train()
            self.worker_network.train()

            # Initialize optimizer
            self.optimizer_manager = torch.optim.SGD(self.manager_network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            self.optimizer_worker = torch.optim.SGD(self.worker_network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

            self.iteration = 0


    # Compute forward pass through manager network to select the object to be packed
    def forward_manager_network(self, input1_selection_network, input2_selection_network, env):

        # Pass input data through model
        chosen_item_index, score_values = self.manager_network.forward(torch.tensor(input1_selection_network, requires_grad=False),torch.tensor(input2_selection_network, requires_grad=False),env.loaded_ids)
        print('----------------------------------------')
        print('Computed Manager network predictions. The object to be packed has been chosen.')

        return chosen_item_index, score_values

    # Compute forward pass through worker network to select the pose of the object to be packed
    def forward_worker_network(self, input_placement_network_1, input_placement_network_2, roll_pitch_angles):

        #-- placement network
        Q_values , [pixel_x,pixel_y,r,p,y],[indices_rp, indices_y] = self.worker_network.forward(torch.tensor(input_placement_network_1, requires_grad=False),torch.tensor(input_placement_network_2, requires_grad=False),roll_pitch_angles)
        print('----------------------------------------')
        print('Computed Worker network predictions. The pose of the object to be packed has been chosen.')

        return  Q_values , [pixel_x,pixel_y,r,p,y],[indices_rp, indices_y]
    
    # Compute target Q_value yt
    def get_Qtarget_value(self, Q_values, prev_obj, current_obj, env):
            # Compute current reward
            current_reward = env.Reward_function(prev_obj, current_obj)

            # Compute expected reward:
            future_reward = float(torch.max(Q_values))

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            yt = current_reward + self.future_reward_discount * future_reward
            print('Expected reward: %f ' % (yt))
            return yt


    # Compute labels and backpropagate
    def backprop(self, Q_values, yt, indices_rp, indices_y, label_weight=1):
            
            # Compute labels to be able to pass gradients only through the selected action (pixel and orientation channel)
            label_Q = torch.zeros_like((Q_values))
            label_Q[:,indices_rp,indices_y,:,:] = yt * label_weight 
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
            self.optimizer_manager.step()
            self.optimizer_worker.step()

