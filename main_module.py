from fonts_terminal import *
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2   
import time 
import pybullet as p
import gc
from trainer import Trainer
from tester import Tester
from env import Env
from experience_replay import ExperienceReplayBuffer
import torch
'''
The main_module.py file is the main script to run the training and testing of the packing problem.
The script is divided into two main functions: train and test.
The train function is used to train the worker network and the manager network.
The test function is used to test the trained networks.
'''
def train(args):

    # Initialize snapshots
    snap = args.snapshot
    snap_targetNN = args.snapshot_targetNet
    
    # Environment setup
    box_size = args.box_size
    resolution = args.resolution
    # list_epochs_for_plot, losses, rewards = [],[],[]

    # Batch size for training
    # batch_size = args.batch_size
    
    # Initialize experience replay buffer 
    replay_buffer = ExperienceReplayBuffer(args.replay_buffer_capacity, args.replay_batch_size)

    for new_episode in range(args.new_episodes):
        # Check if k_min is greater than 2
        if args.k_min < 2:
            raise ValueError("k_min must be greater than or equal to 2")
       
        # Define number of items for current episode
        K_obj = random.choice(list(range(args.k_min,args.k_max+1)))
        k_sort = args.k_sort

        # Initialize episode and epochs counters
        if 'trainer' not in locals():
                
                # First time the main loop is executed
                if args.stage == 1:
                    chosen_train_method = 'stage_1'
                    if args.load_snapshot == True and snap!= None:
                        load_snapshot_ = True
                        episode = int(snap.split('_')[-3])
                        epoch = int(snap.split('_')[-1].strip('.pth'))
                        # sample_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Continuing training after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_ = False
                        episode = 0
                        epoch = 0
                        # sample_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE:", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')

                elif args.stage == 2:
                    chosen_train_method = 'stage_2'
                    if args.load_snapshot == True and snap!= None:
                        load_snapshot_ = True
                        episode = int(snap.split('_')[-3])
                        epoch = int(snap.split('_')[-1].strip('.pth'))
                        # sample_counter = 0                        
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Continuing training after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_ = False
                        episode = 0
                        epoch = 0
                        # sample_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE: ", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                # Initialize trainer (POLICY NET)
                trainer = Trainer(epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay, method = chosen_train_method, future_reward_discount = 0.5, force_cpu = args.force_cpu,
                            load_snapshot = load_snapshot_, file_snapshot = snap,
                            K = k_sort, n_y = args.n_yaw, episode = episode, epoch = epoch)
                print(f"{bold}\nCreo Policy Network{reset}\n") 
                
                # DEFINISCO TARGET NETWORK PER IL CALCOLO DI Q-target
                target_net = Trainer(epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay, method = chosen_train_method, future_reward_discount = 0.5, force_cpu = args.force_cpu,
                            load_snapshot = load_snapshot_, file_snapshot = snap_targetNN,
                            K = k_sort, n_y = args.n_yaw, episode = episode, epoch = epoch)
                target_net.selection_placement_net.load_state_dict(trainer.selection_placement_net.state_dict())
                print(f"{bold}\nCreo Target Network{reset}\n")               

        else:
                # Not the first time the main loop is executed
                episode = episode + 1
                trainer.episode = episode   
                print('----------------------------------------')
                print('----------------------------------------')
                print(f"{purple}NEW EPISODE: ", episode,f"{reset}")                
                print('----------------------------------------')
                print('----------------------------------------')

        # Check if the worker network has already been trained
        print(f"{purple}Worker network already trained on ", epoch, f"EPOCHS{reset}")

        # Initialize environment
        env = Env(obj_dir = args.obj_folder_path, is_GUI = args.gui, box_size=box_size, resolution = resolution)
        print('Set up of PyBullet Simulation environment: \nBox size: ',box_size, '\nResolution: ',resolution)
        
        # Generate csv file with objects
        print('----------------------------------------') 
        K_available = env.generate_urdf_csv()
        print('----------------------------------------')
        
        # Draw Box
        env.draw_box(width=5)

        # Load items 
        item_numbers =  np.random.choice(np.arange(start=0, stop=K_available, step=1), K_obj, replace=True)
        item_ids = env.load_items(item_numbers)
        if len(item_ids) == 0:
            raise ValueError("NO ITEMS LOADED!!")
        
        # Order items by bouding boxes volume
        _, bbox_order = env.order_by_bbox_volume(env.unpacked)

        print('----------------------------------------')
        print(f"{purple_light}K = ", len(bbox_order), 'Items Loaded for this episode', f"{reset}")
        print('--------------------------------------')
        print('Order of objects ids in simulation according to decreasing bounding box volume: ', bbox_order)
        print('--------------------------------------')

        # Initialize variables
        prev_obj = 0 # Objective function  initiaization
        eps_height = 0.05 * box_size[2] # 5% of box height
        tried_obj = []

        # Define the 6 principal views and discretize the roll-pitch angles
        principal_views = {"front": [0, 0, 0],"back": [180, 0, 0],"left": [0, -90, 0],"right": [0, 90, 0],"top": [-90, 0, 0],"bottom": [90, 0, 0]}
        roll, pitch = np.arange(0,360, 360/args.n_rp), np.arange(0,360, 360/args.n_rp) 

        # Initialize variables for the heightmaps at different roll-pitch angles and 6 views
        views = []
        heightmaps_rp = []

        print(f"{bold}--- > Computing inputs to the network.{reset}")

        # If stage 1 is selected the 6 views heightmaps are zero arrays for the k_sort objects with the largest bounding box volume
        if args.stage == 1:
            print('---------------------------------------')
            print(f"{blue_light}\nComputing fake 6 views heightmaps{reset}\n")
            print('---------------------------------------')
            print(f"{blue_light}\nComputing heightmaps at different roll and pitch{reset}\n")
            print('---------------------------------------')

            for i in range(len(list(bbox_order))):
                item_views = []
                for view in principal_views.values():
                    Ht = np.zeros((resolution,resolution))

                    # Uncomment to visualize Heightmaps
                    # env.visualize_object_heightmaps(Ht, Ht, view, only_top = True)
                    # env.visualize_object_heightmaps_3d(Ht, Ht, view, only_top = True)

                    item_views.append(Ht)
                    del(Ht)
                    gc.collect()

                item_views = np.array(item_views)
                views.append(item_views)

                roll_pitch_angles = []      # list of roll-pitch angles
                heightmaps_rp_obj = []      # list of heightmaps for each roll-pitch angle

                for r in roll:
                    for p in pitch:

                        roll_pitch_angles.append(np.array([r,p]))
                        orient = [r,p,0]
                        
                        Ht, Hb, _, _, _ = env.item_hm(bbox_order[i], orient)
                        
                        # Uncomment to visulize Heightmaps
                        #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                        #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                        
                        # add one dimension to concatenate Ht and Hb
                        Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                        Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                        heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                        del(Ht, Hb, orient, _)
                        gc.collect()
                heightmaps_rp.append(heightmaps_rp_obj)

            # If the number of objects is less than k_sort, the remaining objects have zero heightmaps        
            if len(bbox_order) < k_sort:
                for j in range(k_sort-len(bbox_order)):
                    item_views = []
                    for view in principal_views.values():
                        item_views.append(np.zeros((resolution,resolution)))
                    
                    item_views = np.array(item_views)
                    views.append(item_views)
                    
                    heightmaps_rp_obj = []   
                    for r in roll:
                        for p in pitch:

                            orient = [r,p,0]                                
                            Ht, Hb  = np.zeros((resolution,resolution)),np.zeros((resolution,resolution))
                            
                            # --- Uncomment to visulize Heightmaps
                            #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                            #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                            
                            # add one dimension to concatenate Ht and Hb
                            Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                            Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                            heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                            del(Ht, Hb, orient)
                            gc.collect()
                    heightmaps_rp.append(heightmaps_rp_obj)

        elif args.stage == 2:
            print('---------------------------------------')
            print(f"{blue_light}\nComputing fake 6 views heightmaps{reset}\n")
            print('---------------------------------------')
            print(f"{blue_light}\nComputing heightmaps at different roll and pitch{reset}\n")
            print('---------------------------------------')

            for i in range(len(list(bbox_order))):
                item_views = []
                heightmaps_rp_obj = []

                for view in principal_views.values():
                    Ht,_,_,_,_  = env.item_hm(bbox_order[i], view)

                    # Uncomment to visualize Heightmaps
                    # env.visualize_object_heightmaps(Ht, _, view, only_top = True)
                    # env.visualize_object_heightmaps_3d(Ht, _, view, only_top = True)

                    item_views.append(Ht)
                    del(Ht,_)
                    gc.collect()
                item_views = np.array(item_views)
                views.append(item_views)
                roll_pitch_angles = [] # list of roll-pitch angles
                for r in roll:
                    for p in pitch:
                        roll_pitch_angles.append(np.array([r,p]))

                        orient = [r,p,0]

                        #print('Computing heightmaps for object with id: ', next_obj, ' with orientation: ', orient)
                        
                        Ht, Hb, _, _, _ = env.item_hm(bbox_order[i], orient)
                        
                        # Uncomment to visulize Heightmaps
                        # env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                        # env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                        
                        # add one dimension to concatenate Ht and Hb
                        Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                        Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                        heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                        del(Ht, Hb, orient, _)
                        gc.collect()
                heightmaps_rp.append(heightmaps_rp_obj)

            # If the number of objects is less than k_sort, the remaining objects have zero heightmaps        
            if len(bbox_order) < k_sort:
                for j in range(k_sort-len(bbox_order)):
                        item_views = []
                        for view in principal_views.values():
                            item_views.append(np.zeros((resolution,resolution)))
                        item_views = np.array(item_views)
                        views.append(item_views)
                        heightmaps_rp_obj = []
                        for r in roll:
                            for p in pitch:
                                orient = [r,p,0]                                    
                                Ht, Hb  = np.zeros((resolution,resolution)),np.zeros((resolution,resolution))
                                
                                # --- Uncomment to visulize Heightmaps
                                #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                                #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                                
                                # add one dimension to concatenate Ht and Hb
                                Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                                Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                                heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                                del(Ht, Hb, orient)
                                gc.collect()
                        heightmaps_rp.append(heightmaps_rp_obj)
        views, heightmaps_rp = np.array(views), np.asarray(heightmaps_rp) # (K, 6, resolution, resolution)

        # Loop over the loaded objects
        for kk in range(K_obj):
            
            print(f"{purple}Packing iteration for current episode: ", kk, "out of ", K_obj, f"{reset}\n") 
            
            # Compute box heightmap
            heightmap_box = env.box_heightmap()
            print(' --- Computed box Heightmap --- ')

            # Uncomment to visualize heightmap
            # env.visualize_box_heightmap()
            # env.visualize_box_heightmap_3d()

            # Check if there are still items to be packed
            print(' --- Checking if there are still items to be packed --- ')          
            unpacked = env.unpacked
            if len(unpacked) == 0:
                print(f"{bold}{red}NO MORE ITEMS TO PACK --> END OF EPISODE{reset}") 
                snapshot = args.snapshot
                continue
            else:
                print(f"{bold}There are still ", len(unpacked), f" items to be packed.{reset}")
            
            # Check if the box is full
            print(' --- Checking if next item is packable by maximum height --- ')          
            max_Heightmap_box = np.max(heightmap_box)
            is_box_full = max_Heightmap_box > box_size[2] - eps_height
            if is_box_full: 
                print(f"{bold}{red}BOX IS FULL --> END OF EPISODE{reset}")
                snapshot = args.snapshot
                continue
            else:
                print(f"{bold}Max box height not reached yet.{reset}")

            # If the remaining objects are less than k_sort, fill the input tensors with zeros
            if len(bbox_order) < k_sort:
                    for j in range(k_sort-len(bbox_order)):
                        views = np.concatenate((views, np.zeros((1,views.shape[1],resolution,resolution))), axis=0)
                        heightmaps_rp = np.concatenate((heightmaps_rp, np.zeros((1,heightmaps_rp.shape[1],resolution,resolution,heightmaps_rp.shape[-1]))), axis=0)

           
            print('--------------------------------------')
            print('Packed ids before packing: ', env.packed)
            print('UnPacked ids before packing: ', env.unpacked)
            print('--------------------------------------')
            print('Already tried objects: ', tried_obj)
            print('Considering objects with ids: ', bbox_order[0:k_sort], ' for sorting out of: ', bbox_order)
            print('--------------------------------------')

            # Computing the inputs for the network as tensors
            input1_selection_HM_6views = torch.tensor(np.expand_dims(views[0:k_sort], axis=0))                             # (batch, k_sort, 6, resolution, resolution) -- object heightmaps at 6 views
            boxHM = torch.tensor(np.expand_dims(np.expand_dims(heightmap_box,axis=0), axis=0),requires_grad=True)          # (batch, 1, resolution, resolution) -- box heightmap
            input2_selection_ids = torch.tensor([float(item) for item in bbox_order[0:k_sort]] ,requires_grad=True)        # (k_sort) -- list of loaded ids
            input1_placement_rp_angles = torch.tensor(np.asarray(roll_pitch_angles),requires_grad=True)                    # (n_rp, 2) -- roll-pitch angles
            input2_placement_HM_rp = torch.tensor(np.expand_dims(heightmaps_rp[0:k_sort], axis=0),requires_grad=True)      # (batch, k_sort, n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles

            print(f"{blue_light}\nForward pass through the network: Predicting the next object to be packed and the Q values for every candidate pose {reset}\n")
            print('---------------------------------------')           
            Q_values, selected_obj, orients, attention_weights = trainer.forward_network( input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp) # ( n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles
            
            # Update tried objects and remove the selected object from the list of objects to be packed 
            tried_obj.append(selected_obj)
            indices = [i for i, x in enumerate(list(bbox_order)) if x in tried_obj]
            
            # Updates 6views and RPY heightmaps accordingly, filling with zeros the heightmaps of the objects already tried
            zeros = np.zeros_like(views[indices])
            views[indices] = zeros
            non_zero_indices = np.nonzero(views.sum(axis=(1,2,3)))[0]
            zero_indices = np.setdiff1d(np.arange(views.shape[0]), non_zero_indices)
            sorted_indices = np.concatenate((non_zero_indices, zero_indices))
            views = views[sorted_indices]
            
            zeros = np.zeros_like(heightmaps_rp[indices])
            heightmaps_rp[indices] = zeros
            non_zero_indices = np.nonzero(heightmaps_rp.sum(axis=(1,2,3,4)))[0]
            zero_indices = np.setdiff1d(np.arange(heightmaps_rp.shape[0]), non_zero_indices)
            sorted_indices = np.concatenate((non_zero_indices, zero_indices))
            heightmaps_rp = heightmaps_rp[sorted_indices]

            bbox_order = np.array([item for item in list(bbox_order) if item not in tried_obj])

            # Uncomment to plot Q-values
            Qvisual = trainer.visualize_Q_values(Q_values, show=False, save=False, path='snapshots/Q_values/')
            
            print(f"{blue_light}\nChecking placement validity for the best 10 poses {reset}\n")
            indices_rpy, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing, packed, Q_max = trainer.check_placement_validity(env, Q_values, orients, heightmap_box, selected_obj)
            
            # Compute the objective function
            v_items_packed, _ = env.order_by_item_volume(env.packed)
            current_obj = env.Objective_function(env.packed, v_items_packed, env.box_heightmap() , stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
            
            if packed == False:
                print(f"{bold}{red}OBJECT WITH ID: ", selected_obj, f" CANNOT BE PACKED{reset}")
                print('---------------------------------------') 
            
            elif packed == True:                
                # The first iteration does not compute the reward since there are no previous objective function
                if kk>= 1:
                        # Compute reward and Q-target value
                        print('Previous Objective function is: ', prev_obj)
                        print('---------------------------------------') 

                        # copia delle variabili per non creare confusione
                        views_FUTURE = np.copy(views)                  # Copia della variabile views per lo stato futuro
                        heightmaps_rp_FUTURE = np.copy(heightmaps_rp)  # Copia della variabile heightmaps_rp per lo stato futuro

                        # Computing the inputs for the TARGET network as tensors:

                        input1_selection_HM_6views_FUTURE = torch.tensor(np.expand_dims(views_FUTURE[0:k_sort], axis=0))   # CHECK VIEVWS  
                        
                        boxHM_FUTURE = torch.tensor(np.expand_dims(np.expand_dims(NewBoxHeightMap,axis=0), axis=0),requires_grad=True)          # (batch, 1, resolution, resolution) -- box heightmap
                        
                        _, bbox_order_FUTURE = env.order_by_bbox_volume(env.unpacked) #CHECK UNPACKED
                        input2_selection_ids_FUTURE = torch.tensor([float(item) for item in bbox_order_FUTURE[0:k_sort]] ,requires_grad=True)        # (k_sort) -- list of loaded ids
                        
                        input1_placement_rp_angles_FUTURE = torch.tensor(np.asarray(roll_pitch_angles),requires_grad=True)      #CHECK ROLL_PITCH_ANGLES 
                        
                        # If the remaining objects are less than k_sort, fill the input tensors with zeros
                        if len(bbox_order_FUTURE) < k_sort:
                          for j in range(k_sort-len(bbox_order_FUTURE)):
                            views_FUTURE = np.concatenate((views_FUTURE, np.zeros((1,views_FUTURE.shape[1],resolution,resolution))), axis=0)
                            heightmaps_rp_FUTURE = np.concatenate((heightmaps_rp_FUTURE, np.zeros((1,heightmaps_rp_FUTURE.shape[1],resolution,resolution,heightmaps_rp_FUTURE.shape[-1]))), axis=0)
                        input2_placement_HM_rp_FUTURE = torch.tensor(np.expand_dims(heightmaps_rp_FUTURE[0:k_sort], axis=0),requires_grad=True)      # (batch, k_sort, n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles                        

                        #COMPUTE THE CURRENT REWARD
                        current_reward = env.Reward_function(prev_obj, current_obj)
                       
                        # Add the new experience to the replay buffer
                        state = [heightmap_box, input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp]
                        action = [attention_weights, indices_rpy, pixel_x, pixel_y]
                        new_state = [NewBoxHeightMap, input1_selection_HM_6views_FUTURE, boxHM_FUTURE, input2_selection_ids_FUTURE, input1_placement_rp_angles_FUTURE, input2_placement_HM_rp_FUTURE]
                        replay_buffer.add_experience(state, action, current_reward, new_state)


                        # Gradients computation and backpropagation step if the batch size is reached
                        # Extract a (random) bacth of experiences from the buffer 
                        experiences_batch = replay_buffer.sample_batch()

                        Q_values_list = []
                        Q_targets_list = []

                        for experience in experiences_batch:

                            state, action, reward, new_state = experience
                            
                            att_weights, rpy, x, y = action
                            print(f"indices_rpy: {rpy}, type: {type(rpy)}")
                            print(f"pixel_x: {x}, type: {type(x)}")
                            print(f"pixel_y: {y}, type: {type(y)}")
                            print('FATTO1')
                            BoxHeightMap, selection_HM_6views, box_HM, selection_ids, placement_rp_angles, placement_HM_rp = state
                            Qvalues, a, b = trainer.selection_placement_net.placement_net.forward(placement_rp_angles, placement_HM_rp, box_HM, att_weights)
                            Qvalue = Qvalues[rpy, x, y]
                            Q_values_list.append(Qvalue)
                            print('FATTO2')
                            NewBoxHeightmap, selection_HM_6views_FUTURE, box_HM_FUTURE, selection_ids_FUTURE, placement_rp_angles_FUTURE, placement_HM_rp_FUTURE = new_state
                            Qvalues_FUTURE, selected_obj_FUTURE, orients_FUTURE, attention_weights_FUTURE  = target_net.forward_network(selection_HM_6views_FUTURE, box_HM_FUTURE, selection_ids_FUTURE, placement_rp_angles_FUTURE, placement_HM_rp_FUTURE) # ( n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles
                            Qmax_FUTURE = target_net.ebstract_max(Qvalues_FUTURE)    
                            Qtarget = reward + trainer.future_reward_discount * Qmax_FUTURE
                            Q_targets_list.append(Qtarget)
                            print('FATTO3')


                        #  Q_values, selected_obj, orients = self.placement_net(input1_placement_rp_angles, input2_placement_HM_rp, boxHM, attention_weights)
                        #  forward( roll_pitch_angles, input1, input2,  attention_weights):
    

                        # Convert into tensor
                        Q_values_tensor = torch.tensor(Q_values_list, requires_grad=True)
                        if trainer.use_cuda:
                            Q_values_tensor = Q_values_tensor.cuda().float()
                            Q_targets_tensor = torch.tensor(Q_targets_list, requires_grad=True).cuda().float()
                        else:
                            Q_targets_tensor = torch.tensor(Q_targets_list, requires_grad=True).float()

                        Q_targets_tensor = Q_targets_tensor.expand_as(Q_values_tensor)
                        """
                        if self.use_cuda:
                            Q_targets_tensor = torch.tensor(Q_targets_list).cuda().float()
                            Q_targets_tensor = Q_targets_tensor.expand_as(Q_values_tensor)
                        else:
                            Q_targets_tensor = torch.tensor(Q_targets_list).float()
                            Q_targets_tensor = Q_targets_tensor.expand_as(Q_values_tensor)
                        """

                        replay_buffer_length = replay_buffer.get_buffer_length()

                        loss_value = trainer.backprop(Q_targets_tensor, Q_values_tensor, replay_buffer_length, replay_buffer.batch_size)

                        # Update epochs samples counters and save snapshots
                        if replay_buffer_length >= replay_buffer.batch_size:
                            epoch += 1
                            # sample_counter = 0

                            # save and plot losses and rewards
                            trainer.save_and_plot_loss(epoch, loss_value.cpu().detach().numpy(), 'snapshots/losses')
                            trainer.save_and_plot_reward(epoch, current_reward, 'snapshots/rewards')

                            # save snapshots and remove old ones if more than max_snapshots
                            snapshot = trainer.save_snapshot('trainer', max_snapshots=5) 

                            # AGGIORNO TARGET NET
                            if epoch % args.targetNN_freq == 0:
                                target_net.selection_placement_net.load_state_dict(trainer.selection_placement_net.state_dict())
                                snapshot_targetNet = trainer.save_snapshot('targetNet', max_snapshots=5) 
                                print(f"{blue_light}\nAggiorno Target Network {reset}\n")
            
            # Updating the box heightmap and the objective function
            prev_obj = current_obj
            heightmap_box = NewBoxHeightMap
            print(f'\n---------------------------------------') 
            print(f"{bold}{purple}\n -----> PASSING TO THE NEXT OBJECT{reset}\n")
            print('---------------------------------------') 

        del(env)
        gc.collect()

        # AGGIORNO VALORE DI EPSILON ALLA FINE DI OGNI EPISODIO --> trainer.update_epsilon()
        trainer.update_epsilon_exponential()
        print(f"{bold}{blue_light}\n Update epsilon{reset}\n")

    snapshot_targetNet = trainer.save_snapshot('targetNet', max_snapshots=5) 
    print('End of training')

def test(args):
    start_time_ep = time.time()  # Start time
    # Initialize snapshots
    snap = args.snapshot
    
    # Environment setup
    box_size = args.box_size
    resolution = args.resolution
    list_epochs_for_plot, rewards, losses,stability_of_packing_list, latencies = [],[], [], [], []
    compactness_metric, stability_metric, piramidality_metric, n_packed_average = [], [], [], []
    for new_episode in range(args.new_episodes):
        # Check if k_min is greater than 2
        if args.k_min < 2:
            raise ValueError("k_min must be greater than or equal to 2")
       
        # Define number of items for current episode
        K_obj = random.choice(list(range(args.k_min,args.k_max+1)))
        k_sort = args.k_sort

        # Initialize episode and epochs counters
        if 'tester' not in locals():
                
                # First time the main loop is executed
                if args.stage == 1:
                    chosen_test_method = 'stage_1'
                    if args.load_snapshot == True and snap!= None:
                        load_snapshot_ = True
                        episode = int(snap.split('_')[-3])
                        epoch = int(snap.split('_')[-1].strip('.pth'))
                        packed_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Testing after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_ = False
                        episode = 0
                        epoch = 0
                        packed_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE:", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')

                elif args.stage == 2:
                    chosen_test_method = 'stage_2'
                    if args.load_snapshot == True and snap!= None:
                        load_snapshot_ = True
                        episode = int(snap.split('_')[-3])
                        epoch = int(snap.split('_')[-1].strip('.pth'))
                        packed_counter = 0                        
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Continuing testing after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_ = False
                        episode = 0
                        epoch = 0
                        packed_counter = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE: ", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                # Initialize tester
                tester = Tester(method = chosen_test_method, future_reward_discount = 0.5, force_cpu = args.force_cpu,
                            load_snapshot = load_snapshot_, file_snapshot = snap,
                            K = k_sort, n_y = args.n_yaw, episode = episode, epoch = epoch)
        else:
                # Not the first time the main loop is executed
                episode = episode + 1
                tester.episode = episode   
                print('----------------------------------------')
                print('----------------------------------------')
                print(f"{purple}NEW EPISODE: ", episode,f"{reset}")                
                print('----------------------------------------')
                print('----------------------------------------')

        # Check if the worker network has already been trained
        print(f"{purple}Worker network already trained on ", epoch, f"EPOCHS{reset}")

        # Initialize environment
        env = Env(obj_dir = args.obj_folder_path, is_GUI = args.gui, box_size=box_size, resolution = resolution)
        print('Set up of PyBullet Simulation environment: \nBox size: ',box_size, '\nResolution: ',resolution)
        
        # Generate csv file with objects
        print('----------------------------------------') 
        K_available = env.generate_urdf_csv()
        print('----------------------------------------')
        
        # Draw Box
        env.draw_box(width=5)

        # Load items 
        item_numbers =  np.random.choice(np.arange(start=0, stop=K_available, step=1), K_obj, replace=True)
        item_ids = env.load_items(item_numbers)
        if len(item_ids) == 0:
            raise ValueError("NO ITEMS LOADED!!")
        
        # Order items by bouding boxes volume
        _, bbox_order = env.order_by_bbox_volume(env.unpacked)

        print('----------------------------------------')
        print(f"{purple_light}K = ", len(bbox_order), 'Items Loaded for this episode', f"{reset}")
        print('--------------------------------------')
        print('Order of objects ids in simulation according to decreasing bounding box volume: ', bbox_order)
        print('--------------------------------------')

        # Initialize variables
        prev_obj = 0 # Objective function  initiaization
        eps_height = 0.05 * box_size[2] # 5% of box height
        tried_obj = []

        # Define the 6 principal views and discretize the roll-pitch angles
        principal_views = {"front": [0, 0, 0],"back": [180, 0, 0],"left": [0, -90, 0],"right": [0, 90, 0],"top": [-90, 0, 0],"bottom": [90, 0, 0]}
        roll, pitch = np.arange(0,360, 360/args.n_rp), np.arange(0,360, 360/args.n_rp) 

        # Initialize variables for the heightmaps at different roll-pitch angles and 6 views
        views = []
        heightmaps_rp = []

        print(f"{bold}--- > Computing inputs to the network.{reset}")

        # If stage 1 is selected the 6 views heightmaps are zero arrays for the k_sort objects with the largest bounding box volume
        if args.stage == 1:
            print('---------------------------------------')
            print(f"{blue_light}\nComputing fake 6 views heightmaps{reset}\n")
            print('---------------------------------------')
            print(f"{blue_light}\nComputing heightmaps at different roll and pitch{reset}\n")
            print('---------------------------------------')

            for i in range(len(list(bbox_order))):
                item_views = []
                for view in principal_views.values():
                    Ht = np.zeros((resolution,resolution))

                    # Uncomment to visualize Heightmaps
                    # env.visualize_object_heightmaps(Ht, Ht, view, only_top = True)
                    # env.visualize_object_heightmaps_3d(Ht, Ht, view, only_top = True)

                    item_views.append(Ht)
                    del(Ht)
                    gc.collect()

                item_views = np.array(item_views)
                views.append(item_views)

                roll_pitch_angles = []      # list of roll-pitch angles
                heightmaps_rp_obj = []      # list of heightmaps for each roll-pitch angle

                for r in roll:
                    for p in pitch:

                        roll_pitch_angles.append(np.array([r,p]))
                        orient = [r,p,0]
                        
                        Ht, Hb, _, _, _ = env.item_hm(bbox_order[i], orient)
                        
                        # Uncomment to visulize Heightmaps
                        #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                        #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                        
                        # add one dimension to concatenate Ht and Hb
                        Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                        Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                        heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                        del(Ht, Hb, orient, _)
                        gc.collect()
                heightmaps_rp.append(heightmaps_rp_obj)

            # If the number of objects is less than k_sort, the remaining objects have zero heightmaps        
            if len(bbox_order) < k_sort:
                for j in range(k_sort-len(bbox_order)):
                    item_views = []
                    for view in principal_views.values():
                        item_views.append(np.zeros((resolution,resolution)))
                    
                    item_views = np.array(item_views)
                    views.append(item_views)
                    
                    heightmaps_rp_obj = []   
                    for r in roll:
                        for p in pitch:

                            orient = [r,p,0]                                
                            Ht, Hb  = np.zeros((resolution,resolution)),np.zeros((resolution,resolution))
                            
                            # --- Uncomment to visulize Heightmaps
                            #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                            #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                            
                            # add one dimension to concatenate Ht and Hb
                            Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                            Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                            heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                            del(Ht, Hb, orient)
                            gc.collect()
                    heightmaps_rp.append(heightmaps_rp_obj)

        elif args.stage == 2:
            print('---------------------------------------')
            print(f"{blue_light}\nComputing fake 6 views heightmaps{reset}\n")
            print('---------------------------------------')
            print(f"{blue_light}\nComputing heightmaps at different roll and pitch{reset}\n")
            print('---------------------------------------')

            for i in range(len(list(bbox_order))):
                item_views = []
                heightmaps_rp_obj = []

                for view in principal_views.values():
                    Ht,_,_,_,_  = env.item_hm(bbox_order[i], view)

                    # Uncomment to visualize Heightmaps
                    # env.visualize_object_heightmaps(Ht, _, view, only_top = True)
                    # env.visualize_object_heightmaps_3d(Ht, _, view, only_top = True)

                    item_views.append(Ht)
                    del(Ht,_)
                    gc.collect()
                item_views = np.array(item_views)
                views.append(item_views)
                roll_pitch_angles = [] # list of roll-pitch angles
                for r in roll:
                    for p in pitch:
                        roll_pitch_angles.append(np.array([r,p]))

                        orient = [r,p,0]

                        #print('Computing heightmaps for object with id: ', next_obj, ' with orientation: ', orient)
                        
                        Ht, Hb, _, _, _ = env.item_hm(bbox_order[i], orient)
                        
                        # Uncomment to visulize Heightmaps
                        # env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                        # env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                        
                        # add one dimension to concatenate Ht and Hb
                        Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                        Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                        heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                        del(Ht, Hb, orient, _)
                        gc.collect()
                heightmaps_rp.append(heightmaps_rp_obj)

            # If the number of objects is less than k_sort, the remaining objects have zero heightmaps        
            if len(bbox_order) < k_sort:
                for j in range(k_sort-len(bbox_order)):
                        item_views = []
                        for view in principal_views.values():
                            item_views.append(np.zeros((resolution,resolution)))
                        item_views = np.array(item_views)
                        views.append(item_views)
                        heightmaps_rp_obj = []
                        for r in roll:
                            for p in pitch:
                                orient = [r,p,0]                                    
                                Ht, Hb  = np.zeros((resolution,resolution)),np.zeros((resolution,resolution))
                                
                                # --- Uncomment to visulize Heightmaps
                                #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                                #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                                
                                # add one dimension to concatenate Ht and Hb
                                Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                                Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                                heightmaps_rp_obj.append(np.concatenate((Ht,Hb), axis=2)) 
                                del(Ht, Hb, orient)
                                gc.collect()
                        heightmaps_rp.append(heightmaps_rp_obj)
        views, heightmaps_rp = np.array(views), np.asarray(heightmaps_rp) # (K, 6, resolution, resolution)

        # Loop over the loaded objects
        for kk in range(K_obj):
            
            print(f"{purple}Packing iteration for current episode: ", kk, "out of ", K_obj, f"{reset}\n") 
            
            # Compute box heightmap
            heightmap_box = env.box_heightmap()
            print(' --- Computed box Heightmap --- ')

            # Uncomment to visualize heightmap
            # env.visualize_box_heightmap()
            # env.visualize_box_heightmap_3d()

            # Check if there are still items to be packed
            print(' --- Checking if there are still items to be packed --- ')          
            unpacked = env.unpacked
            if len(unpacked) == 0:
                print(f"{bold}{red}NO MORE ITEMS TO PACK --> END OF EPISODE{reset}") 
                snapshot = args.snapshot
                continue
            else:
                print(f"{bold}There are still ", len(unpacked), f" items to be packed.{reset}")
            
            # Check if the box is full
            print(' --- Checking if next item is packable by maximum height --- ')          
            max_Heightmap_box = np.max(heightmap_box)
            is_box_full = max_Heightmap_box > box_size[2] - eps_height
            if is_box_full: 
                print(f"{bold}{red}BOX IS FULL --> END OF EPISODE{reset}")
                snapshot = args.snapshot
                continue
            else:
                print(f"{bold}Max box height not reached yet.{reset}")

            # If the remaining objects are less than k_sort, fill the input tensors with zeros
            if len(bbox_order) < k_sort:
                    for j in range(k_sort-len(bbox_order)):
                        views = np.concatenate((views, np.zeros((1,views.shape[1],resolution,resolution))), axis=0)
                        heightmaps_rp = np.concatenate((heightmaps_rp, np.zeros((1,heightmaps_rp.shape[1],resolution,resolution,heightmaps_rp.shape[-1]))), axis=0)

           
            print('--------------------------------------')
            print('Packed ids before packing: ', env.packed)
            print('UnPacked ids before packing: ', env.unpacked)
            print('--------------------------------------')
            print('Already tried objects: ', tried_obj)
            print('Considering objects with ids: ', bbox_order[0:k_sort], ' for sorting out of: ', bbox_order)
            print('--------------------------------------')

            # Computing the inputs for the network as tensors
            input1_selection_HM_6views = torch.tensor(np.expand_dims(views[0:k_sort], axis=0))                             # (batch, k_sort, 6, resolution, resolution) -- object heightmaps at 6 views
            boxHM = torch.tensor(np.expand_dims(np.expand_dims(heightmap_box,axis=0), axis=0),requires_grad=True)          # (batch, 1, resolution, resolution) -- box heightmap
            input2_selection_ids = torch.tensor([float(item) for item in bbox_order[0:k_sort]] ,requires_grad=True)        # (k_sort) -- list of loaded ids
            input1_placement_rp_angles = torch.tensor(np.asarray(roll_pitch_angles),requires_grad=True)                    # (n_rp, 2) -- roll-pitch angles
            input2_placement_HM_rp = torch.tensor(np.expand_dims(heightmaps_rp[0:k_sort], axis=0),requires_grad=True)      # (batch, k_sort, n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles

            print(f"{blue_light}\nForward pass through the network: Predicting the next object to be packed and the Q values for every candidate pose {reset}\n")
            print('---------------------------------------')    
            start_time = time.time()  # Start time       
            Q_values, selected_obj, orients = tester.forward_network( input1_selection_HM_6views, boxHM, input2_selection_ids, input1_placement_rp_angles, input2_placement_HM_rp) # ( n_rp, res, res, 2) -- object heightmaps at different roll-pitch angles
            end_time = time.time()  # Start time
            latency = end_time - start_time  # Calculate latency
            print(f"Latency for tester.forward_network(): {latency:.6f} seconds")
            latencies.append(latency)  # Append latency to the list
            # Update tried objects and remove the selected object from the list of objects to be packed 
            tried_obj.append(selected_obj)
            indices = [i for i, x in enumerate(list(bbox_order)) if x in tried_obj]
            
            # Updates 6views and RPY heightmaps accordingly, filling with zeros the heightmaps of the objects already tried
            zeros = np.zeros_like(views[indices])
            views[indices] = zeros
            non_zero_indices = np.nonzero(views.sum(axis=(1,2,3)))[0]
            zero_indices = np.setdiff1d(np.arange(views.shape[0]), non_zero_indices)
            sorted_indices = np.concatenate((non_zero_indices, zero_indices))
            views = views[sorted_indices]
            
            zeros = np.zeros_like(heightmaps_rp[indices])
            heightmaps_rp[indices] = zeros
            non_zero_indices = np.nonzero(heightmaps_rp.sum(axis=(1,2,3,4)))[0]
            zero_indices = np.setdiff1d(np.arange(heightmaps_rp.shape[0]), non_zero_indices)
            sorted_indices = np.concatenate((non_zero_indices, zero_indices))
            heightmaps_rp = heightmaps_rp[sorted_indices]

            bbox_order = np.array([item for item in list(bbox_order) if item not in tried_obj])

            # Uncomment to plot Q-values
            Qvisual = tester.visualize_Q_values(Q_values, show=False, save=True, path='snapshots/Q_values_test/')
            
            print(f"{blue_light}\nChecking placement validity for the best 10 poses {reset}\n")
            indices_rpy, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing, packed, Q_max = tester.check_placement_validity(env, Q_values, orients, heightmap_box, selected_obj)
            stability_of_packing_list.append(stability_of_packing)
            # Compute the objective function
            v_items_packed, _ = env.order_by_item_volume(env.packed)
            current_obj = env.Objective_function(env.packed, v_items_packed, env.box_heightmap() , stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
            
            if packed == False:
                print(f"{bold}{red}OBJECT WITH ID: ", selected_obj, f" CANNOT BE PACKED{reset}")
                print('---------------------------------------') 
                
            elif packed == True: 
                # Count the number of packed samples 
                packed_counter += 1
                print(f'{red}\nRecorded ', packed_counter, f' packed items.{reset}')
                                                  
                # The first iteration does not compute the reward since there are no previous objective function
                if kk>= 1:
                        # Compute reward and Q-target value
                        print('Previous Objective function is: ', prev_obj)
                        print('---------------------------------------')           
                        current_reward, Q_target = tester.get_Qtarget_value(Q_max, prev_obj, current_obj, env)
                        loss_value = tester.loss(Q_values, Q_target, indices_rpy, pixel_x, pixel_y)

                        # save and plot losses and rewards
                        list_epochs_for_plot.append(epoch)
                        rewards.append(current_reward)
                        losses.append(loss_value.cpu().detach().numpy())
                        tester.save_and_plot_reward(list_epochs_for_plot, rewards, 'snapshots/rewards_test')
                        tester.save_and_plot_loss(list_epochs_for_plot, losses, 'snapshots/losses_test')

            
            # Updating the box heightmap and the objective function
            prev_obj = current_obj
            heightmap_box = NewBoxHeightMap
            print(f'\n---------------------------------------') 
            print(f"{bold}{purple}\n -----> PASSING TO THE NEXT OBJECT{reset}\n")
            print('---------------------------------------') 

        end_time_ep = time.time()  # Start time
        # Compute compactness and piramidality
        v_items_packed, _ = env.order_by_item_volume(env.packed)
        compactness, piramidality = env.Compactness(env.packed, v_items_packed, env.box_heightmap() ), env.Pyramidality(env.packed, v_items_packed, env.box_heightmap() )
        # Check if the list is not empty
        if stability_of_packing_list:
            # Compute the average
            average_stability = sum(stability_of_packing_list) / len(stability_of_packing_list)
        else:
            average_stability = 0
            print('The stability_of_packing_list is empty.')
        
        # Append the metrics to the lists    
        compactness_metric.append(compactness) 
        stability_metric.append(average_stability)
        piramidality_metric.append(piramidality)
        n_packed_average.append(packed_counter)
        
        print('---------------------------------------') 
        print(f'{red_light}Compactness: ', compactness, f'{reset}')
        print(f'{red_light}Piramidality: ', piramidality, f'{reset}')
        print(f'{red_light}N packed items:',packed_counter, f'{reset}')
        print(f'{red_light}Stability: ',average_stability, f'{reset}')
        print(f'{red_light}Episode duration: ',end_time_ep - start_time_ep, f'{reset}')
        print('---------------------------------------') 
        print(f'{red}END OF CURRENT EPISODE: ', episode, f'{reset}')
        
        del(env)
        gc.collect()
        
    average_stability_global = sum(stability_metric) / len(stability_metric)
    average_compactness_global = sum(compactness_metric) / len(compactness_metric)
    average_piramidality_global = sum(piramidality_metric) / len(piramidality_metric)
    average_n_packed_global = sum(n_packed_average) / len(n_packed_average)
    average_latency = sum(latencies) / len(latencies)
    
    print('---------------------------------------')
    print(f'{red}Average Compactness: ', average_compactness_global, f'{reset}')
    print(f'{red}Average Piramidality: ', average_piramidality_global, f'{reset}')
    print(f'{red}Average Stability: ', average_stability_global, f'{reset}')
    print(f'{red}Average N packed items:', average_n_packed_global, f'{reset}')
    print(f'{red}Average Latency: ', average_latency, f'{reset}')
    print('---------------------------------------')
    print('End of testing')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser for training')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_folder_path',  action='store', default='objects/easy_setting/') # path to the folder containing the objects .csv file
    parser.add_argument('--gui', dest='gui', action='store', default=False) # GUI for PyBullet
    parser.add_argument('--force_cpu', dest='force_cpu', action='store', default=False) # Use CPU instead of GPU
    parser.add_argument('--stage', action='store', default=1) # stage 1 or 2 for training
    parser.add_argument('--k_max', action='store', default=5) # max number of objects to load
    parser.add_argument('--k_min', action='store', default=5) # min number of objects to load
    parser.add_argument('--k_sort', dest='k_sort', action='store', default=5) # number of objects to consider for sorting
    parser.add_argument('--resolution', dest='resolution', action='store', default=50) # resolution of the heightmaps
    parser.add_argument('--box_size', dest='box_size', action='store', default=(0.4,0.4,0.3)) # size of the box
    parser.add_argument('--snapshot', dest='snapshot', action='store', default=f'snapshots/models/network_episode_39_epoch_7.pth') # path to the  network snapshot
    parser.add_argument('--snapshot_targetNet', dest='snapshot_targetNet', action='store', default=f'snapshots/models/targetNet/network_episode_47_epoch_10.pth') # path to the target network snapshot
    parser.add_argument('--new_episodes', action='store', default=40) # number of episodes
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store', default=False) # Load snapshot 
    # parser.add_argument('--batch_size', dest='batch_size', action='store', default=70) # Batch size for training
    parser.add_argument('--n_yaw', action='store', default=2) # 360/n_y = discretization of yaw angle
    parser.add_argument('--n_rp', action='store', default=2)  # 360/n_rp = discretization of roll and pitch angles
    
    # epsilon-greedy parameters: 
    parser.add_argument('--epsilon', action='store', default=0.9)          # Valore iniziale per epsilon
    parser.add_argument('--epsilon_min', action='store', default=0.05)     # Valore minimo per epsilon
    parser.add_argument('--epsilon_decay', action='store', default=0.995)   # Fattore di decrescita per epsilon

    # frequenza di aggiornamento della target network
    parser.add_argument('--targetNN_freq', action='store', default=12)          # target network aggiornata ogni N epoche (i pesi della policy network copiati su target network)

    # experience replay
    parser.add_argument('--replay_buffer_capacity', action='store', default=30) #size of the experience replay buffer 
    parser.add_argument('--replay_batch_size', action='store', default=2) #size of the batch ebstracted from experience replay buffer

    args = parser.parse_args()
    
    # --------------- Start Train --------------- 153 epochs stage 1 
    train(args) 
     # --------------- Start Test ---------------   NOT ready yet
    #test(args)

