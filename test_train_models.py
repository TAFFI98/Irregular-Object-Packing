from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')
from models import conv_block, Upsample, Downsample, final_conv_select_net, placement_net, selection_net
import torch
import pybullet as p
from env import Env
import numpy as np
from scipy.ndimage import rotate
from trainer import Trainer
import cv2
'''
This file starts by setting up the PyBullet environment and generating a CSV file with object information. 
Then, it loads a specified number of random objects into the environment and simulates the physics of the objects.
The script uses a trainer object to train and evaluate a packing algorithm. 
The packing algorithm has two stages: stage 1 and stage 2. 
In stage 1, the script computes the bounding box volumes of the loaded objects and orders them based on their volumes. 
In stage 2, the script computes heightmaps for the loaded objects from six different views and feeds them into a selection network to choose the best object to pack.
For each object to be packed, the script discretizes the roll and pitch angles and computes heightmaps for each combination of roll and pitch angles. 
These heightmaps are then fed into a placement network to determine the best placement position for the object. 
The script checks the validity of the predicted placement position by considering collisions with the box margins and the height of the box.
After each object is packed, the script computes an objective function that evaluates the quality of the packing. 
The script uses the objective function to compute a reward and performs backpropagation to update the weights of the placement network.
The script repeats this process for a specified number of iterations, packing objects one by one. 
'''
if __name__ == '__main__':
        

    #-- Path with the URDF files
    obj_folder_path = '/Project/Irregular-Object-Packing/objects/'
    
    #-- PyBullet Environment setup 
    box_size=(0.4,0.4,0.3)
    resolution = 50
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=box_size, resolution = resolution)
    print('----------------------------------------')
    print('Setup of PyBullet environment: \nBox size: ',box_size, '\nResolution: ',resolution)

    #-- Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()
    print('----------------------------------------')
    print('Generating CSV with objects')

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
    K = 8
    item_numbers = np.random.randint(0, 100, size=K)
    item_ids = env.load_items(item_numbers)

    for i in range(500):
        p.stepSimulation()
    print('----------------------------------------')
    print('K = ', K, 'Items Loaded')
    print('--------------------------------------')
    print('Loaded ids before packing: ', env.loaded_ids)
    print('Packed ids before packing: ', env.packed)
    print('UnPacked ids before packing: ', env.unpacked)
    print('--------------------------------------')
    prev_obj = 0

    # Initialize trainer
    trainer = Trainer(method= 'stage_2', future_reward_discount=0.5,
                      is_testing='False', load_snapshot=False, snapshot_file='',
                      force_cpu='False',K=K)
    print('--------------------------------------')
    print('Trainer initialized') 

    #-- Compute bounding boxes  Volumes and orders them  - for stage 1
    volume_bbox, bbox_order = env.order_by_bbox_volume(env.loaded_ids)
    print(' The order by bbox volume is: ', bbox_order)

    method = 'stage_1' # stage_1 or stage_2

    for k in range(K):
        #-- Compute Box HeightMap: shape (resolution, resolution)
        BoxHeightMap = env.box_heightmap()
        print('----------------------------------------')
        print('Computed Box Height Map')
        
        #-- Compute the 6 views heightmaps for alle the loaded objects and store them in a numpy array
        if method == 'stage_2':
            principal_views = {
                "front": [0, 0, 0],
                "back": [180, 0, 0],
                "left": [0, -90, 0],
                "right": [0, 90, 0],
                "top": [-90, 0, 0],
                "bottom": [90, 0, 0]
            }
            six_views_all_items = []
            for id_ in env.loaded_ids:
                six_views_one_item = []
                for view in principal_views.values():
                    if id_ in env.unpacked:
                        Ht,_ = env.item_hm(id_, view)
                        #env.visualize_object_heightmaps(id_, view, only_top = True )
                        #env.visualize_object_heightmaps_3d(id_, view, only_top = True )
                        six_views_one_item.append(Ht) 
                    if id_ in env.packed:  
                        Ht = np.zeros((resolution,resolution))
                        six_views_one_item.append(Ht)               
                six_views_one_item = np.array(six_views_one_item)
                six_views_all_items.append(six_views_one_item)

            six_views_all_items = np.array(six_views_all_items) # (K, 6, resolution, resolution)
            

            print('----------------------------------------')
            print('Computed Heightmaps of all the loaded objects with the 6 views')
        
            # to account for batch = 1 and channels of bounding box heightmap
            input1_selection_network = np.expand_dims(six_views_all_items, axis=0)  #(batch, K, 6, resolution, resolution)
            input2_selection_network = np.expand_dims(np.expand_dims(BoxHeightMap,axis=0), axis=0)  #(batch, 1, resolution, resolution)
            print('----------------------------------------')
            print('Computed inputs for Manager Selection network')
            #-- FORWARD MANAGER NETWORK 
            chosen_item_index, score_values = trainer.forward_manager_network(input1_selection_network,input2_selection_network, env)
        
        if method == 'stage_1':

            chosen_item_index = bbox_order[k]

        
        #-- Discretize roll, pitch and yaw
        print('----------------------------------------')
        print('Discretizing roll and pitch angles')
        roll_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
        pitch_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi

        item_heightmaps_RP = []
        roll_pitch_angles = []
        for r,roll in enumerate(roll_angles):
                for p,pitch in enumerate(pitch_angles):
                    orient = [roll, pitch, 0]

                    roll_pitch_angles.append(np.array([roll,pitch]))
                    Ht,Hb = env.item_hm(chosen_item_index, orient )
                    #env.visualize_object_heightmaps(chosen_item_index, orient)
                    #env.visualize_object_heightmaps_3d(chosen_item_index,orient )
                    Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                    Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                    item_heightmap = np.concatenate((Ht, Hb), axis=2)
                    item_heightmaps_RP.append(item_heightmap)  
        print('----------------------------------------')
        print('Computed Bottom and Top Heightmaps for alle the roll and pitch angles')

        item_heightmaps_RP = np.asarray(item_heightmaps_RP)  #(16, res, res, 2)
        roll_pitch_angles =  np.asarray(roll_pitch_angles)   #(16, 2) ---> [roll,pitch]
        num_rp = item_heightmaps_RP.shape[0]
        n_y = 16
        # Placement networks inputs
        input_placement_network_1 = item_heightmaps_RP
        input_placement_network_2 = np.expand_dims(BoxHeightMap, axis=2) #(res, res, 1)
    
        # to accounto for batch = 1
        input_placement_network_1 = np.expand_dims(input_placement_network_1, axis=0) #(res, res, 1)
        input_placement_network_2 = np.expand_dims(input_placement_network_2, axis=0) #(batch, res, res, 1)
        print('----------------------------------------')
        print('Computed inputs for Worker Placement network')
        #-- FORWARD WORKER NETWORK    
        Q_values = trainer.forward_worker_network(input_placement_network_1, input_placement_network_2, roll_pitch_angles)
        

        # Show the Q values
        Q_values_visual = trainer.visualize_Q_values(Q_values)    
        cv2.namedWindow("Q Values", cv2.WINDOW_NORMAL)
        cv2.imshow("Q Values", Q_values_visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Check if the predicted pose is allowed: Collision with box margins and exceeded height of the box
        indices_rp, indices_y, pixel_x, pixel_y, NewBoxHeightMap, stability_of_packing = trainer.check_placement_validity(env, Q_values, roll_pitch_angles, BoxHeightMap, chosen_item_index)
        
        
        print('Packing item ', k+1, 'with id: ', chosen_item_index)

        
        # Cmpute objective function
        volume_items_packed, _ = env.order_by_item_volume(env.packed)
        current_obj = env.Objective_function(env.packed, volume_items_packed, BoxHeightMap, stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
        if i>= 1:
            ''' Compute reward '''
            print('Previous Objective function is: ', prev_obj)
            print('Backpropagating...')
            yt = trainer.get_Qtarget_value(Q_values, indices_rp, indices_y, pixel_x, pixel_y, prev_obj, current_obj, env)
            trainer.backprop(Q_values, yt, indices_rp, indices_y, pixel_x, pixel_y, label_weight=1)

        prev_obj = current_obj
        BoxHeightMap = NewBoxHeightMap
    
 






