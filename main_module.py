from fonts_terminal import *
import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2    
import pybullet_data
import pybullet as p

from trainer import Trainer
from tester import Tester
from env import Env
import logging
from logger import Logger
import utils

def train(args):

    # Initialize snapshots
    manager_snap = args.manager_snapshot
    worker_snap = args.worker_snapshot
    list_epochs_for_plot, losses, rewards = [],[],[]
    
    for new_episode in range(args.new_episodes):
        
        # Initialize episode and epochs counters
        if 'trainer' not in locals():
                
                # First time the main loop is executed
                if args.stage == 1:
                    chosen_train_method = 'stage_1'
                    load_snapshot_manager_ = False
                    if args.load_snapshot_worker == True and worker_snap!= None:
                        load_snapshot_worker_ = True
                        episode = int(worker_snap.split('_')[-3])
                        epoch = int(worker_snap.split('_')[-1].strip('.pth'))
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Continuing training after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_worker_ = False
                        episode = 0
                        epoch = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE:", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')

                elif args.stage == 2:
                    chosen_train_method = 'stage_2'
                    if args.load_snapshot_worker == True and worker_snap!= None:
                        load_snapshot_worker_ = True
                        episode = int(worker_snap.split('_')[-3])
                        epoch = int(worker_snap.split('_')[-1].strip('.pth'))
                        epoch = int(worker_snap.split('_')[-1].strip('.pth'))
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Continuing training after", episode, f" episodes already simulated{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    else:
                        load_snapshot_worker_ = False
                        episode = 0
                        epoch = 0
                        print('----------------------------------------')
                        print('----------------------------------------')
                        print(f"{purple}Starting from scratch --> EPISODE: ", episode,f"{reset}")                
                        print('----------------------------------------')
                        print('----------------------------------------')
                    if args.load_snapshot_manager == True and manager_snap!= None:
                        load_snapshot_manager_ = True
                    else:   
                        load_snapshot_manager_ = False


        else:
                # Not the first time the main loop is executed
                episode = new_episode
                print('----------------------------------------')
                print('----------------------------------------')
                print(f"{purple}NEW EPISODE: ", episode,f"{reset}")                
                print('----------------------------------------')
                print('----------------------------------------')
                

        # Initialize trainer
        trainer = Trainer(method = chosen_train_method, future_reward_discount = 0.5, force_cpu = args.force_cpu,
                load_snapshot_manager = load_snapshot_manager_, load_snapshot_worker = load_snapshot_worker_, 
                file_snapshot_manager = manager_snap, file_snapshot_worker = worker_snap,
                K=args.k_sort, episode = episode, epoch = epoch)

        # Environment setup
        box_size = (0.4,0.4,0.3)
        resolution = 50
        env = Env(obj_dir = args.obj_folder_path, is_GUI = args.gui, box_size=box_size, resolution = resolution)
        print('Set up of PyBullet Simulation environment: \nBox size: ',box_size, '\nResolution: ',resolution)


        # Define number of items for current episode
        if args.k_min < 2:
            raise ValueError("k_min must be greater than or equal to 2")
        K_obj = random.choice(list(range(args.k_min,args.k_max+1)))
        # Generate csv file with objects
        print('----------------------------------------') 
        _ = env.generate_urdf_csv()
        print('----------------------------------------')
        print(f'{bold}Generating CSV with ', K_obj, f' objects:{reset}')
        #-- Draw Box
        env.draw_box(width=5)

        #-- Load items 
        values = np.arange(start=0, stop=100+1, step=1)
        #item_numbers = np.array([17,19]) #to debug
        item_numbers =  np.random.choice(values, K_obj, replace=False)
        
        item_ids = env.load_items(item_numbers)

        # volumes, sorted_ids = env.order_by_item_volume(item_ids)
        _, bbox_order = env.order_by_bbox_volume(item_ids)
        for i in range(500):
            env.p.stepSimulation() 

        print('----------------------------------------')
        print(f"{purple_light}K = ", K_obj, 'Items Loaded for this episode', f"{reset}")
        print('--------------------------------------')
        print('Order of objects ids in simulation according to decreasing bounding box volume: ', bbox_order)
        print('Loaded ids before packing: ', env.loaded_ids)
        print('Packed ids before packing: ', env.packed)
        print('UnPacked ids before packing: ', env.unpacked)
        print('--------------------------------------')

        prev_obj = 0 #objective function score initiaization
        eps_height = 0.05 * box_size[2] # 5% of box height
        print(f"{purple}Worker network already trained on ", epoch, f"EPOCHS{reset}")
        # loop over the loaded objects
        for i in range(K_obj):
            print(f"{purple}Packing iteration for current episode: ", i, "out of ", K_obj, f"{reset}") 
            heightmap_box = env.box_heightmap()
            print(' --- Computed box Heightmap --- ')
            #env.visualize_box_heightmap()
            #env.visualize_box_heightmap_3d()

            print(' --- Checking if there are still items to be packed --- ')          
            unpacked = env.unpacked
            if len(unpacked) == 0:
                print(f"{bold}{red}NO MORE ITEMS TO PACK --> END OF EPISODE{reset}") 
                continue
            else:
                print(f"{bold}There are still ", len(unpacked), f" items to be packed.{reset}")

            print(' --- Checking if next item is packable by maximum height --- ')          
            max_Heightmap_box = np.max(heightmap_box)
            is_box_full = max_Heightmap_box > box_size[2] - eps_height
            if is_box_full: 
                print(f"{bold}{red}BOX IS FULL --> END OF EPISODE{reset}")
                continue
            else:
                print(f"{bold}Max box height not reached yet.{reset}")

            
            # selection strategy according to the chosen stage
            if args.stage == 1:
                # pack largest object first
                print('---------------------------------------')
                print(f"{bold}\n1. SELECTION: Object selection according to largest bounding box volume{reset}")
                next_obj = bbox_order[i]
                print("Selected object id: ", next_obj)

            elif args.stage == 2:

                # select k objects with the largest bounding box volume
                print('---------------------------------------')
                print(f"{yellow}\n1. SELECTION: Object selection according to Manager Network Selection{reset}")
                
                k_sort = args.k_sort 

                principal_views = {
                        "front": [0, 0, 0],
                        "back": [180, 0, 0],
                        "left": [0, -90, 0],
                        "right": [0, 90, 0],
                        "top": [-90, 0, 0],
                        "bottom": [90, 0, 0]
                    }
                
                views = []

                # if number of object is less than k, update k       
                if len(env.unpacked) < k_sort:
                    k_sort = len(env.unpacked)

                unpacked_volume, unpacked_ids = env.order_by_bbox_volume(env.unpacked)

                # select k objects with the largest boubding box volume, compute thier heightmaps and concatenate them            
                for i in range(k_sort):
                    item_views = []

                    for view in principal_views.values():
                        Ht,_,obj_length,obj_width, _  = env.item_hm(unpacked_ids[i], view)
                        #env.visualize_object_heightmaps(Ht, _, view, only_top = True)
                        #env.visualize_object_heightmaps_3d(Ht, _, view, only_top = True)
                        item_views.append(Ht)

                    item_views = np.array(item_views)
                    views.append(item_views)

                views = np.array(views) # (K, 6, resolution, resolution)

                # forward pass in the manager network to get the scores for next object to be packed
                input1_selection_network = np.expand_dims(views, axis=0)  #(batch, K, 6, resolution, resolution)
                input2_selection_network = np.expand_dims(np.expand_dims(heightmap_box,axis=0), axis=0)  #(batch, 1, resolution, resolution)

                chosen_item_index, score_values = trainer.forward_manager_network(input1_selection_network,input2_selection_network, env)

                next_obj = chosen_item_index

            # discretize r,p,y angles --> 90 degrees discretization
            roll = np.arange(0,360,90) 
            pitch = np.arange(0,360,90) 
            roll_pitch_angles = [] # list of roll-pitch angles
            heightmaps_rp = []     # list of heightmaps for each roll-pitch angle

            for r in roll:
                for p in pitch:

                    roll_pitch_angles.append(np.array([r,p]))
                    orient = [r,p,0]

                    #print('Computing heightmaps for object with id: ', next_obj, ' with orientation: ', orient)
                    
                    Ht, Hb, _, _, _ = env.item_hm(next_obj, orient)
                    
                    # --- Uncomment to visulize Heightmaps
                    #env.visualize_object_heightmaps(Ht, Hb, orient, only_top = False)
                    #env.visualize_object_heightmaps_3d(Ht, Hb, orient, only_top = False)
                    
                    # add one dimension to concatenate Ht and Hb
                    Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                    Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                    heightmaps_rp.append(np.concatenate((Ht,Hb), axis=2)) 

            heightmaps_rp = np.asarray(heightmaps_rp) # (16, res, res, 2)
            roll_pitch_angles = np.asarray(roll_pitch_angles) # (16, 2)

            # prepare inputs for forward pass in the worker network
            input_placement_network_1 = np.expand_dims(heightmaps_rp, axis=0)                          # (batch, 16, res, res, 2) -- object heightmaps at different roll-pitch angles
            input_placement_network_2 = np.expand_dims(np.expand_dims(heightmap_box, axis=2), axis =0) # (batch, res, res, 1)     -- box heightmap

            # forward pass into worker to get Q-values for placement
            Q_values = trainer.forward_worker_network(env, roll_pitch_angles, input_placement_network_1, input_placement_network_2)
            print(f"{bold}\n2. PLACEMENT: Computing packing pose of the selected object with placement network{reset}")
            print("Q values computed for every candidate pose.\n")

            # Uncomment to plot Q-values
            # Qvisual = trainer.visualize_Q_values(Q_values, show=True)


            # check placement validity
            indices_rp, indices_y, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing = trainer.check_placement_validity(env, Q_values, roll_pitch_angles, heightmap_box, next_obj)

            # Compute objective function  
            v_items_packed, _ = env.order_by_item_volume(env.packed)
            current_obj = env.Objective_function(env.packed, v_items_packed, NewBoxHeightMap, stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
            if i>= 1:
                    ''' Compute reward '''
                    print('Previous Objective function is: ', prev_obj)
                    current_reward, yt = trainer.get_Qtarget_value(Q_values, indices_rp, indices_y, pixel_x, pixel_y, prev_obj, current_obj, env)
                    
                    loss_value = trainer.backprop(Q_values, yt, indices_rp, indices_y, pixel_x, pixel_y, label_weight=1)
                
                    # Updating training epoch
                    epoch += 1
                    print(f"{purple}Worker network trained on ", epoch, f"EPOCHS{reset}")
                    
                    # save snapshots and remove old ones if more than max_snapshots
                    manager_snapshot, worker_snapshot = trainer.save_snapshot(max_snapshots=5) 
                    
                    # save and plot losses and rewards
                    list_epochs_for_plot.append(epoch)
                    losses.append(loss_value)
                    rewards.append(current_reward)
                    trainer.save_and_plot_loss(list_epochs_for_plot, losses, 'snapshots/losses')
                    trainer.save_and_plot_reward(list_epochs_for_plot, rewards, 'snapshots/rewards')
            
            prev_obj = current_obj
            heightmap_box = NewBoxHeightMap


        manager_snap = manager_snapshot
        worker_snap = worker_snapshot

        print('--------------------------------------')
        print(f'{red}END OF CURRENT EPISODE: ', episode, f'{reset}')
        print('--------------------------------------')

    print('End of training')

def test(args):

    # Setup environment
    #is_sim = args.is_sim        # per adesso is_sim lo lascio fisso a True, poi da definire a livello di logica quando si implementa il robot fisico

    # environment setup
    box_size=(0.4,0.4,0.3)
    resolution = 50
    env = Env(obj_dir = args.obj_folder_path, is_GUI=True, box_size=box_size, resolution = resolution)
    print('----------------------------------------')
    print('Setup of PyBullet environment: \nBox size: ',box_size, '\nResolution: ',resolution)

    # Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()
    print('----------------------------------------')
    print('Generating CSV with objects')

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
   
    item_numbers = np.random.randint(0, 100, size=K)
    item_ids = env.load_items(item_numbers)

    # volumes, sorted_ids = env.order_by_item_volume(item_ids)
    volume_bbox, bbox_order = env.order_by_bbox_volume(item_ids)

    # initialize tester
    tester = Tester(file_snapshot_manager='snapshots/models/manager_network_1.pth',file_snapshot_worker='snapshots/models/worker_network_1.pth', force_cpu=False, K=args.k_obj)

    print('--------------------------------------')
    print('Tester initialized')

    # main for loop for testing
    for i in range(args.k_obj):

        heightmap_box = env.box_heightmap()
        eps_height = 0.05 * box_size[2] # 5% of box height

        average_stability = []

        if np.size(env.unpacked) == 0:

            volume_items_packed, _ = env.order_by_item_volume(env.packed)

            print('------------------------- METRICS -------------------------')
            pyramidality = env.Pyramidality(env.packed, volume_items_packed, NewBoxHeightMap)
            compactness = env.Compactness(env.packed, volume_items_packed, NewBoxHeightMap)
            stability = np.mean(average_stability)
            print('Piramidality: ', pyramidality)  
            print('Compactness: ', compactness)
            print('Stability: ', stability)
            print('Number of packed items: ', len(env.packed))

        else:
            max_Heightmap_box = np.max(heightmap_box)
            is_box_full = max_Heightmap_box > box_size[2] - eps_height #valutare se sostituire con funzione chje guarda altezza di tutte le celle della heightmap
            
            if  is_box_full:

                volume_items_packed, _ = env.order_by_item_volume(env.packed)

                print('------------------------- METRICS -------------------------')
                pyramidality = env.Pyramidality(env.packed, volume_items_packed, NewBoxHeightMap)
                compactness = env.Compactness(env.packed, volume_items_packed, NewBoxHeightMap)
                stability = np.mean(average_stability)
                print('Piramidality: ', pyramidality)  
                print('Compactness: ', compactness)
                print('Stability: ', stability)
                print('Number of packed items: ', len(env.packed))

                print('Box is full' )
                break
                
            else:

                # select k_sort objects with largest bbox volume
                unpacked_volume, unpacked_ids = env.order_by_bbox_volume(env.unpacked)

                # select k-sort objects with largest bbox volume, compute their heightmaps for k_sort * 6 views and concatenate with box heightmap
                k_sort = args.k_sort
                #k_sort_ids = unpacked_ids[:k_sort]

                principal_views = {
                    "front": [0, 0, 0],
                    "back": [180, 0, 0],
                    "left": [0, -90, 0],
                    "right": [0, 90, 0],
                    "top": [-90, 0, 0],
                    "bottom": [90, 0, 0]
                }

                views = []

                # if number of object is less than k, update k       
                if len(env.unpacked) < k_sort:
                    k_sort = len(env.unpacked)

                for i in range(k_sort):
                    item_views = []

                    for view in principal_views.values():
                        Ht,_,_,_ ,_ = env.item_hm(unpacked_ids[i], view)
                        #env.visualize_object_heightmaps(id_, view, only_top = True )
                        #env.visualize_object_heightmaps_3d(id_, view, only_top = True )
                        item_views.append(Ht)

                    item_views = np.array(item_views)
                    views.append(item_views)    

                views = np.array(views) # (K, 6, resolution, resolution) 

                # prepare inputs for forward pass in the manager network to get the scores for next object to be packed
                input1_selection_network = np.expand_dims(views, axis=0)  #(batch, K, 6, resolution, resolution)
                input2_selection_network = np.expand_dims(np.expand_dims(heightmap_box,axis=0), axis=0)  #(batch, 1, resolution, resolution)
                # forward pass in the manager network
                chosen_item_index, score_values = tester.forward_manager_network(input1_selection_network,input2_selection_network, env)

                next_obj = chosen_item_index

                # discretize r,p,y angles
                roll = np.arange(0,360,90) # 90 degrees discretization
                pitch = np.arange(0,360,90)

                roll_pitch_angles = [] # list of roll-pitch angles
                hiehgtmaps_rp = [] # list of heightmaps for each roll-pitch angle

                for r in roll:
                    for p in pitch:

                        roll_pitch_angles.append(np.array([r,p]))

                        orient = [r,p,0]
                        Ht, Hb, _, _ ,_= env.item_hm(next_obj, orient)
                        
                        # add one dimension to concatenate Ht and Hb
                        Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                        Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                        heightmaps_rp.append(np.concatenate((Ht,Hb), axis=2))

                heightmaps_rp = np.asarray(heightmaps_rp) # (16, res, res, 2)
                roll_pitch_angles = np.asarray(roll_pitch_angles) # (16, 2)

                # prepare inputs for forward pass in the worker network
                input_placement_network_1 = np.expand_dims(heightmaps_rp, axis=0)  # (batch, 16, res, res, 2)
                input_placement_network_2 = np.expand_dims(np.expand_dims(heightmap_box, axis=2), axis =0) #(batch, res, res, 1)

                Q_values = tester.forward_worker_network(input_placement_network_1, input_placement_network_2, roll_pitch_angles)
                print('Q values computed')

                # check placement validity and update heightmap
                indices_rp, indices_y, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing = tester.check_placement_validity(env, Q_values, roll_pitch_angles, heightmap_box, next_obj)
                average_stability.append(stability_of_packing)
                print('Packing item with id: ', next_obj)
                heightmap_box = NewBoxHeightMap                 

    print('Testing done')

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser for training')

    # --------------- Setup options ---------------
    parser.add_argument('--is_testing', dest='is_testing', action='store', default=False)
    parser.add_argument('--obj_folder_path',  action='store', default='objects/') # path to the folder containing the objects .csv file
    parser.add_argument('--gui', dest='gui', action='store', default=True) # GUI for PyBullet
    parser.add_argument('--force_cpu', dest='force_cpu', action='store', default=False) # Use CPU instead of GPU
    parser.add_argument('--stage', action='store', default=1) # stage 1 or 2 for training
    parser.add_argument('--k_max', action='store', default=13) # max number of objects to load
    parser.add_argument('--k_min', action='store', default=10) # min number of objects to load
    parser.add_argument('--k_sort', dest='k_sort', action='store', default=2) # number of objects to consider for sorting
    parser.add_argument('--manager_snapshot', dest='manager_snapshot', action='store', default=f'snapshots/models/worker_network_episode_0_epoch_1.pth') # path to the manager network snapshot
    parser.add_argument('--worker_snapshot', dest='worker_snapshot', action='store', default=f'snapshots/models/worker_network_episode_7_epoch_10.pth') # path to the worker network snapshot
    parser.add_argument('--new_episodes', action='store', default=10000) # number of episodes
    parser.add_argument('--load_snapshot_manager', dest='load_snapshot_manager', action='store', default=False) # Load snapshot of the manager network
    parser.add_argument('--load_snapshot_worker', dest='load_snapshot_worker', action='store', default=False) # Load snapshot of the worker network

    args = parser.parse_args()
    
    # --------------- Start Train ---------------
    train(args) 
     # --------------- Start Test ---------------   
    #test(args)
