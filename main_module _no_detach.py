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
from models import placement_net
from models import selection_net
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
    snap_sel = args.snapshot_sel
    snap_pla = args.snapshot_pla
    snap_targetSEL = args.snapshot_targetNet_sel
    snap_targetPLA = args.snapshot_targetNet_pla
    
    # Environment setup
    box_size = args.box_size
    resolution = args.resolution
    
    # Initialize experience replay buffer (per la PLACEMENT NET)
    replay_buffer = ExperienceReplayBuffer(args.replay_buffer_capacity, args.replay_batch_size)
    counter_threshold_pla = args.sample_counter_threshold_placement_net
    
    # SELECTION NET aggiornata ogni 4 epoche delle PLACEMENT NET (come nel codice originale)
    epoch_threshold_sel = args.epoch_counter_threshold_selection_net

    # Define max number of attempts to pack an item:
    max_attempts = 5

    for new_episode in range(args.new_episodes):
        # Check if k_min is greater than 2
        if args.k_min < 2:
            raise ValueError("k_min must be greater than or equal to 2")
       
        # Define number of items for current episode
        K_obj = random.choice(list(range(args.k_min,args.k_max+1)))
        k_sort = args.k_sort

        # Initialize episode and epochs counters
        
        # First time the main loop is executed:
        if 'policy_sel_net' not in locals():
            if args.load_snapshot == True and snap_sel!= None and snap_pla!= None:
                load_snapshot_ = True
                episode = int(snap_sel.split('_')[-3]) +1                       # aggiunto +1 così conteggio degli episodi è consistente
                epoch_sel = int(snap_sel.split('_')[-1].strip('.pth'))
                epoch_pla = int(snap_pla.split('_')[-1].strip('.pth') )
                sample_counter = 0                        
                print('----------------------------------------')
                print('----------------------------------------')
                print(f"{purple}Continuing training after", episode-1 , f" episodes already simulated{reset}")                
                print('----------------------------------------')
                print('----------------------------------------')
            else:
                load_snapshot_ = False
                episode = 0
                epoch_sel = 0
                epoch_pla = 0
                sample_counter = 0
                print('----------------------------------------')
                print('----------------------------------------')
                print(f"{purple}Starting from scratch --> EPISODE: ", episode,f"{reset}")                
                print('----------------------------------------')
                print('----------------------------------------')
            
            # Initialize trainer (POLICY NET)
            print(f"{bold}\nCreo Policy Networks{reset}\n") 
            print('---------------------------------------------------')
            print(f"{bold}creo Selection Network [MANAGER]{reset}\n") 
            policy_sel_net = Trainer('manager', epsilon=args.epsilon_sel, epsilon_min=args.epsilon_min_sel, epsilon_decay=args.epsilon_decay_sel, 
                                    future_reward_discount = 0.5, force_cpu = args.force_cpu,
                                    load_snapshot = load_snapshot_, file_snapshot = snap_sel,
                                    K = k_sort, n_y = args.n_yaw, episode = episode, epoch = epoch_sel)
            print('---------------------------------------------------')
            print(f"{bold}creo Placement Network [WORKER]{reset}\n") 
            policy_pla_net = Trainer('worker', epsilon=args.epsilon_pla, epsilon_min=args.epsilon_min_pla, epsilon_decay=args.epsilon_decay_pla, 
                                    future_reward_discount = 0.5, force_cpu = args.force_cpu,
                                    load_snapshot = load_snapshot_, file_snapshot = snap_pla,
                                    K = k_sort, n_y = args.n_yaw, episode = episode, epoch = epoch_pla)
            print('---------------------------------------------------')
            print(f"{bold}\nCreo Target Networks{reset}")
            target_sel_net = Trainer('manager', epsilon=args.epsilon_sel, epsilon_min=args.epsilon_min_sel, epsilon_decay=args.epsilon_decay_sel, 
                                    future_reward_discount = 0.5, force_cpu = args.force_cpu,
                                    load_snapshot = load_snapshot_, file_snapshot = snap_targetSEL,
                                    K = k_sort, n_y = args.n_yaw, episode = 0, epoch = 0)
            target_pla_net = Trainer('worker', epsilon=args.epsilon_pla, epsilon_min=args.epsilon_min_pla, epsilon_decay=args.epsilon_decay_pla, 
                                    future_reward_discount = 0.5, force_cpu = args.force_cpu,
                                    load_snapshot = load_snapshot_, file_snapshot = snap_targetPLA,
                                    K = k_sort, n_y = args.n_yaw, episode = 0, epoch = 0)
            
            # inizializzo le target nets con stessi pesi delle policy nets (se episodio 0, non vengono caricati snapshots)
            if load_snapshot_ == False:
                print(f"{red}{bold}\nCopio pesi Policy Networks su Target Networks{reset}\n") 
                target_sel_net.selection_net.load_state_dict(policy_sel_net.selection_net.state_dict())
                target_pla_net.placement_net.load_state_dict(policy_pla_net.placement_net.state_dict())
    
        # If not the first time the main loop is executed    
        else:
            episode = episode + 1
            policy_sel_net.episode = episode   
            policy_pla_net.episode = episode
            print('----------------------------------------')
            print('----------------------------------------')
            print(f"{purple}NEW EPISODE: ", episode,f"{reset}")                
            print('----------------------------------------')
            print('----------------------------------------')

        # Check if the networks has already been trained
        print(f"{purple}Manager network already trained on ", epoch_sel, f"EPOCHS{reset}")
        print(f"{purple}Worker network already trained on ", epoch_pla, f"EPOCHS{reset}")

        # Initialize environment
        env = Env(obj_dir = args.obj_folder_path, is_GUI = args.gui, box_size=box_size, resolution = resolution)
        print('\nSet up of PyBullet Simulation environment: \nBox size: ',box_size, '\nResolution: ',resolution)
        
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
                snapshot_sel = args.snapshot_sel
                snapshot_pla = args.snapshot_pla
                break
                # continue
            else:
                print(f"{bold}There are still ", len(unpacked), f" items to be packed.{reset}")
            
            # Check if the box is full
            print(' --- Checking if next item is packable by maximum height --- ')          
            max_Heightmap_box = np.max(heightmap_box)
            is_box_full = max_Heightmap_box > box_size[2] - eps_height
            if is_box_full: 
                print(f"{bold}{red}BOX IS FULL --> END OF EPISODE{reset}")
                snapshot_sel = args.snapshot_sel
                snapshot_pla = args.snapshot_pla
                break
                # continue
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

            # FORWARD PASS attraverso la selection net e la placement net
            print(f"{blue_light}\nForward pass through the network: Predicting the next object to be packed and the Q values for every candidate pose {reset}\n")
            print('---------------------------------------')           
            
            # Forward Selection Net
            Q_values_sel, attention_weights = policy_sel_net.selection_net.forward( input1_selection_HM_6views, boxHM, input2_selection_ids) 

            #PRENDE OGGETTO DA attention_weights
            # EXPLOITATION EXPLORATION TRADE-OFF: EPSILON-GREEDY
            if np.random.rand() < policy_sel_net.epsilon:
                # Scegli un'azione casuale
                print(f'{red_light}Sto eseguendo EXPLORATION!{reset}') 
                while True:  # Loop fino a trovare un oggetto non inserito
                  sel_obj = torch.randint(0, len(Q_values_sel[0]), (1,)).item()  # Seleziona casualmente
                  if Q_values_sel[0][sel_obj].item() != 0:  # Controlla se l'oggetto è già inserito
                    break 
            else:
                # Scegli l'azione con il massimo Q-value
                print(f'{red_light}Sto eseguendo EXPLOITATION!{reset}') 
                sel_obj = int(torch.argmax(Q_values_sel).cpu().numpy())

            selected_obj = int(input2_selection_ids.clone().cpu().detach()[sel_obj]) 
            
            print(sel_obj)
            print(selected_obj)
            
            Qvalue_sel = Q_values_sel[:, sel_obj]

            print('Qvalues DELLA SELECTION NET:')
            print(Q_values_sel)

            # Forwad Placement Net
            copy_boxHM = boxHM.clone().detach()
            with torch.no_grad():
                Q_values_pla, orients = policy_pla_net.placement_net.forward(input1_placement_rp_angles, input2_placement_HM_rp, copy_boxHM, attention_weights.detach()) 

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
            Qvisual_pla = policy_pla_net.visualize_Q_values(Q_values_pla, show=False, save=False, path='snapshots/placeemnt_net/Q_values_sel/')
            Qvisual_sel = policy_sel_net.visualize_Q_values(Q_values_sel, show=False, save=False, path='snapshots/selection_net/Q_values_pla/')
            
            print(f"{blue_light}\nChecking placement validity for the best {max_attempts} poses {reset}\n")
            #indices_rpy, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing, packed, attempt = policy_pla_net.check_placement_validity(env, Q_values_pla, orients, heightmap_box, selected_obj)
            indices_rpy, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing, packed, attempt = policy_pla_net.check_placement_validity_TENTATIVES(env, Q_values_pla, orients, heightmap_box, selected_obj, max_attempts)
            
            sample_counter += 1
            print(f'{red}\nRecorded ', sample_counter, f' samples for 1 batch of training{reset}')
                
            # Compute the objective function
            v_items_packed, _ = env.order_by_item_volume(env.packed)
            current_obj = env.Objective_function(packed, env.packed, v_items_packed, env.box_heightmap() , stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
            if packed == False:
                print(f"{bold}{red}OBJECT WITH ID: ", selected_obj, f" CANNOT BE PACKED{reset}")
                print('---------------------------------------') 
            
            elif packed == True:                
                print(f"{bold}{green}OBJECT WITH ID: ", selected_obj, f"IS PACKED{reset}")
                print('---------------------------------------') 
            
            # The first iteration does not compute the reward since there are no previous objective function
            if kk>= 1:
                # Compute reward and Q-target value
                print('Previous Objective function is: ', prev_obj)
                print('---------------------------------------') 

                # copia delle variabili per non creare confusione
                views_FUTURE = np.copy(views)                  # Copia della variabile views per lo stato futuro
                heightmaps_rp_FUTURE = np.copy(heightmaps_rp)  # Copia della variabile heightmaps_rp per lo stato futuro

                # Computing the inputs for the TARGET networks as tensors:
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

                #COMPUTE THE REWARD per il placement
                reward_pla = env.calculate_reward(packed, attempt, max_attempts)
                
                # Add the new experience to the replay buffer
                state = [copy_boxHM, input1_placement_rp_angles, input2_placement_HM_rp]
                action = [attention_weights.detach(), indices_rpy, pixel_x, pixel_y]
                current_reward = reward_pla
                new_state = [input1_selection_HM_6views_FUTURE, boxHM_FUTURE, input2_selection_ids_FUTURE, input1_placement_rp_angles_FUTURE, input2_placement_HM_rp_FUTURE]
                replay_buffer.add_experience(state, action, current_reward, new_state)
                
                # Extract a (random) bacth of experiences from the buffer 
                experiences_batch = replay_buffer.sample_batch()

                Q_values_list = []
                Q_targets_list = []

                for experience in experiences_batch:
                    state, action, reward, new_state = experience
                    att_weights, rpy, x, y = action
                    box_HM, placement_rp_angles, placement_HM_rp = state                
                    selection_HM_6views_FUTURE, box_HM_FUTURE, selection_ids_FUTURE, placement_rp_angles_FUTURE, placement_HM_rp_FUTURE = new_state

                    # CALCOLO Qvalues della PLACEMENT net
                    Q_values_pla, _ = policy_pla_net.placement_net.forward(placement_rp_angles, placement_HM_rp, box_HM, att_weights) 
                    Qval_pla = Q_values_pla[rpy, x, y]

                    # verifica se episodio prelevato coincide con un TERMINAL STATE
                    if torch.any(selection_ids_FUTURE):
                        # forward pass attraverso la TARGET Selection Net
                        _, attention_weights_FUTURE = target_sel_net.selection_net.forward(selection_HM_6views_FUTURE, box_HM_FUTURE, selection_ids_FUTURE) 
                        # forward pass attraverso la TARGET Placeemnt Net
                        Qvalues_pla_FUTURE, _ = target_pla_net.placement_net.forward(placement_rp_angles_FUTURE, placement_HM_rp_FUTURE, box_HM_FUTURE, attention_weights_FUTURE) 
                        Qmax_FUTURE = target_pla_net.ebstract_max(Qvalues_pla_FUTURE)    
                    else:
                        Qmax_FUTURE = 0
                    Qtarget_pla = reward + policy_pla_net.future_reward_discount * Qmax_FUTURE
                     
                    Qtar_pla_tensor = torch.tensor(Qtarget_pla).float()
                    Qtar_pla_tensor = Qtar_pla_tensor.expand_as(Qval_pla)
                    Q_targets_list.append(Qtar_pla_tensor)
                
                    Q_values_list.append(Qval_pla)
                

                # Convert into tensors
                if policy_pla_net.use_cuda:
                    Qval_tensor_pla = torch.stack(Q_values_list).cuda()
                    Qtar_tensor_pla = torch.stack(Q_targets_list).cuda()
                else:
                    Qval_tensor_pla = torch.stack(Q_values_list)
                    Qtar_tensor_pla = torch.stack(Q_targets_list)

                # Verifica che i tensori abbiano le dimensioni giuste
                assert Qval_tensor_pla.size() == Qtar_tensor_pla.size()
                # Verifica che i tensori abbiano lo stesso tipo di dato
                assert Qval_tensor_pla.dtype == Qtar_tensor_pla.dtype
        
                # Update epochs samples counters and save snapshots
                if replay_buffer.get_buffer_length() >= replay_buffer.batch_size and sample_counter % counter_threshold_pla == 0:    
                    # Calcola la loss e esegui backpropagation (se necessario)
                    loss_value_pla = policy_pla_net.backprop_pla(Qtar_tensor_pla, Qval_tensor_pla, replay_buffer.get_buffer_length(), replay_buffer.batch_size, sample_counter, counter_threshold_pla)
                    
                    epoch_pla += 1
                    sample_counter = 0
                    
                    # save and plot losses and rewards
                    policy_pla_net.save_and_plot_loss(epoch_pla, loss_value_pla.cpu().detach().numpy(), 'snapshots/placement_net/losses')
                    policy_pla_net.save_and_plot_reward(epoch_pla, reward_pla, 'snapshots/placement_net/rewards')

                    # AGGIORNO TARGET NET e salvo snapshot
                    if epoch_pla % args.target_pla_freq == 0:
                        target_pla_net.placement_net.load_state_dict(policy_pla_net.placement_net.state_dict())
                        snapshot_target_pla = target_pla_net.save_snapshot('worker', 'snapshots/placement_net/target', max_snapshots=5) 
                        print(f"{red}{bold}Aggiorno Target Placement Network {reset}")

                    # save snapshots and remove old ones if more than max_snapshots
                    if epoch_pla % 10 == 0: 
                        snapshot = policy_pla_net.save_snapshot('worker','snapshots/placement_net/trainer', max_snapshots=5) 

                # Calcolo Reward per il Manager (proviene dall'environment)
                reward_sel = env.Reward_function(prev_obj, current_obj)
                # CALCOLO Qtarget della PLACEMENT net
                    
                # Forward TARGET selection net
                if torch.any(selection_ids_FUTURE):
                    Q_values_sel_FUTURE, _  = target_sel_net.selection_net.forward(selection_HM_6views_FUTURE, box_HM_FUTURE, selection_ids_FUTURE) 
                    Qmax_FUTURE = target_sel_net.ebstract_max(Q_values_sel_FUTURE)    
                else:
                    Qmax_FUTURE = 0
                Q_target_sel = reward + policy_sel_net.future_reward_discount * Qmax_FUTURE

                # Convert into tensor
                if policy_sel_net.use_cuda:
                    Qtar_tensor_sel = torch.tensor(Q_target_sel).cuda().float()
                    Qtar_tensor_sel = Qtar_tensor_sel.expand_as(Qvalue_sel)
                else:
                    Qtar_tensor_sel = torch.tensor(Q_target_sel).float()
                    Qtar_tensor_sel = Qtar_tensor_sel.expand_as(Qvalue_sel)

                
                if sample_counter == 0 and policy_pla_net.epoch % epoch_threshold_sel == 0:    
                    # calcolo della loss ed esegui bacpropagation
                    loss_value_sel = policy_sel_net.backprop_sel(Qtar_tensor_sel, Qvalue_sel, policy_pla_net.epoch, epoch_threshold_sel)
                    
                    epoch_sel += 1
                    
                    # save and plot losses and rewards
                    policy_sel_net.save_and_plot_loss(epoch_sel, loss_value_sel.cpu().detach().numpy(), 'snapshots/selection_net/losses')
                    policy_sel_net.save_and_plot_reward(epoch_sel, reward_sel, 'snapshots/selection_net/rewards')

                    # Aggiorno TARGET selection net
                    if epoch_sel % args.target_sel_freq == 0:
                        target_sel_net.selection_net.load_state_dict(policy_sel_net.selection_net.state_dict())
                        snapshot_target_sel = target_sel_net.save_snapshot('manager','snapshots/selection_net/target', max_snapshots=5) 
                        print(f"{red}{bold}Aggiorno Target selection Network {reset}")

                    # save snapshots and remove old ones if more than max_snapshots
                    if epoch_sel % 10 == 0: 
                        snapshot = policy_sel_net.save_snapshot('manager', 'snapshots/selection_net/trainer', max_snapshots=5) 

            # Updating the box heightmap and the objective function
            prev_obj = current_obj
            heightmap_box = NewBoxHeightMap
            print(f'\n---------------------------------------') 
            print(f"{bold}{purple}\n -----> PASSING TO THE NEXT OBJECT{reset}\n")
            print('---------------------------------------') 

        del(env)
        gc.collect()

        # AGGIORNO VALORE DI EPSILON ALLA FINE DI OGNI EPISODIO --> trainer.update_epsilon()
        if episode+1 % 3 == 0:
          policy_sel_net.update_epsilon_exponential()

        policy_pla_net.update_epsilon_exponential()

    snapshot_sel = policy_sel_net.save_snapshot('manager', 'snapshots/selection_net/trainer', max_snapshots=5) 
    snapshot_pla = policy_pla_net.save_snapshot('worker', 'snapshots/placement_net/trainer', max_snapshots=5) 
    print(f"{red}{bold}salvo Policy Networks {reset}\n")

    snapshot_target_pla = target_sel_net.save_snapshot('manager', 'snapshots/selection_net/target', max_snapshots=5) 
    snapshot_target_sel = target_pla_net.save_snapshot('worker', 'snapshots/placement_net/target', max_snapshots=5) 
    print(f"{red}{bold}salvo Target Networks {reset}\n")

    print('End of training')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser for training')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_folder_path',  action='store', default='objects/easy_setting/') # path to the folder containing the objects .csv file
    parser.add_argument('--gui', dest='gui', action='store', default=False) # GUI for PyBullet
    parser.add_argument('--force_cpu', dest='force_cpu', action='store', default=False) # Use CPU instead of GPU
    parser.add_argument('--stage', action='store', default=1) # stage 1 or 2 for training
    parser.add_argument('--k_max', action='store', default=3) # max number of objects to load
    parser.add_argument('--k_min', action='store', default=3) # min number of objects to load
    parser.add_argument('--k_sort', dest='k_sort', action='store', default=3) # number of objects to consider for sorting
    parser.add_argument('--resolution', dest='resolution', action='store', default=50) # resolution of the heightmaps
    parser.add_argument('--box_size', dest='box_size', action='store', default=(0.4,0.4,0.3)) # size of the box
    parser.add_argument('--snapshot_sel', dest='snapshot_sel', action='store', default=f'snapshots/selectionNet/models/policyNet/network_episode_3292_epoch_494.pth') # path to the  network snapshot
    parser.add_argument('--snapshot_targetNet_sel', dest='snapshot_targetNet_sel', action='store', default=f'snapshots/selectionNet/models/targetNet/network_episode_0_epoch_492.pth') # path to the target network snapshot
    parser.add_argument('--snapshot_pla', dest='snapshot_pla', action='store', default=f'snapshots/placementNet/models/policyNet/network_episode_3292_epoch_494.pth') # path to the  network snapshot
    parser.add_argument('--snapshot_targetNet_pla', dest='snapshot_targetNet_pla', action='store', default=f'snapshots/placementNet/models/targetNet/network_episode_0_epoch_492.pth') # path to the target network snapshot
    parser.add_argument('--new_episodes', action='store', default=5) # number of episodes
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store', default=False) # Load snapshot     parser.add_argument('--n_yaw', action='store', default=2) # 360/n_y = discretization of yaw angle
    parser.add_argument('--n_yaw', action='store', default=2) # 360/n_y = discretization of yaw angle
    parser.add_argument('--n_rp', action='store', default=2)  # 360/n_rp = discretization of roll and pitch angles
    
    # epsilon-greedy parameters: 
    parser.add_argument('--epsilon_sel', action='store', default=0.975)           # Valore iniziale per epsilon
    parser.add_argument('--epsilon_min_sel', action='store', default=0.03)      # Valore minimo per epsilon
    parser.add_argument('--epsilon_decay_sel', action='store', default=0.9985)   # Fattore di decrescita per epsilon
     
    parser.add_argument('--epsilon_pla', action='store', default=0.975)           # Valore iniziale per epsilon
    parser.add_argument('--epsilon_min_pla', action='store', default=0.05)      # Valore minimo per epsilon
    parser.add_argument('--epsilon_decay_pla', action='store', default=0.9975)   # Fattore di decrescita per epsilon

    # frequenza di aggiornamento della target network
    parser.add_argument('--target_sel_freq', action='store', default=2)          # target network aggiornata ogni N epoche (i pesi della policy network copiati su target network)
    parser.add_argument('--target_pla_freq', action='store', default=2)          # target network aggiornata ogni N epoche (i pesi della policy network copiati su target network)

    # experience replay
    parser.add_argument('--replay_buffer_capacity', action='store', default=10)                 #size of the experience replay buffer 
    parser.add_argument('--replay_batch_size', action='store', default=2)                        #size of the batch ebstracted from experience replay buffer
    parser.add_argument('--sample_counter_threshold_placement_net', action='store', default=3)  #dopo quanti inseriemnti viene aggiornato il worker

    parser.add_argument('--epoch_counter_threshold_selection_net', action='store', default=2)   #dopo quante epoche viene aggiornato il manager

    args = parser.parse_args()
    
    # --------------- Start Train --------------- 
    train(args) 
     # --------------- Start Test ---------------   
    #test(args)