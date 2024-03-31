from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')
from models import conv_block, Upsample, Downsample, final_conv_select_net, placement_net, selection_net
import torch
import pybullet as p
from env import Env
import numpy as np
from scipy.ndimage import rotate

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
    K = 3
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
    for k in range(K):
        #-- Compute Box HeightMap: shape (resolution, resolution)
        BoxHeightMap = env.box_heightmap()
        print('----------------------------------------')
        print('Computed Box Height Map')
        
        #-- Compute the 6 views heightmaps for alle the loaded objects and store them in a numpy array

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
        print('Computed inputs for Manager Sleection network')
        #-- selection network
        print('----------------------------------------')
        print('Initialized manager network')
        manager_network = selection_net(use_cuda=False,K = K)
        chosen_item_index, score_values = manager_network(torch.tensor(input1_selection_network),torch.tensor(input2_selection_network), env.loaded_ids)
        print('----------------------------------------')
        print('Computed Manager network predictions. The object to be packed has been chosen.')

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
        
        # Placement networks inputs
        input_placement_network_1 = item_heightmaps_RP
        input_placement_network_2 = np.expand_dims(BoxHeightMap, axis=2) #(res, res, 1)
    
        # to accounto for batch = 1
        input_placement_network_1 = np.expand_dims(input_placement_network_1, axis=0) #(res, res, 1)
        input_placement_network_2 = np.expand_dims(input_placement_network_2, axis=0) #(batch, res, res, 1)
        print('----------------------------------------')
        print('Computed inputs for Worker Placement network')
        #-- placement_net visulize (too big it crashes)
        #placement_net_graph = draw_graph(placement_net(use_cuda=False), input_data=[torch.tensor(input_placement_network_1),torch.tensor(input_placement_network_2)],graph_name = 'placement_net',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/')
        #placement_net_graph.visual_graph

        worker_network = placement_net(use_cuda=False)
        print('----------------------------------------')
        print('Initialized worker network')
        Q_values , [pixel_x,pixel_y,r,p,y] = worker_network(torch.tensor(input_placement_network_1),torch.tensor(input_placement_network_2),roll_pitch_angles)
        print('----------------------------------------')
        print('Computed Worker network predictions. The pose of the object to be packed has been chosen.')

        print('----------------------------------------')
        print('Packing chosen item with predicted pose...')
        # Pack chosen item with predicted pose
        target_euler = [r,p,y]
        target_pos = [pixel_x* box_size[0]*100/resolution,pixel_y * box_size[1]*100/resolution,0] # cm
        transform = np.empty(6,)
        transform[0:3] = target_euler
        transform[3:6] = target_pos
        BoxHeightMap, stability_of_packing = env.pack_item(chosen_item_index , transform)
        print('--------------------------------------')
        print('Packed chosen item ')
        print('Is the placement stable?', stability_of_packing)
        print('Loaded ids after packing: ', env.loaded_ids)
        print('Packed ids after packing: ', env.packed)
        print('UnPacked ids after packing: ', env.unpacked)
        print('--------------------------------------')
        volume_items_packed, _ = env.order_by_item_volume(env.packed)
        current_obj = env.Objective_function(env.packed, volume_items_packed, BoxHeightMap, stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
        if i>= 1:
            ''' Compute reward '''
            print('Previous Objective function is: ', prev_obj)
            Reward = env.Reward_function(prev_obj, current_obj)
        prev_obj = current_obj
    
    """ 
    Visualize rotated Heightmaps 
    yaw_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    num_yaws = yaw_angles.shape[0]
    item_heightmaps_RPY = np.empty(shape=(num_yaws, num_rp, item_heightmaps_RP.shape[1],item_heightmaps_RP.shape[2], 2)) # shape: (num_yaws, num_RP, resolution, resolution, 2)

    for j,yaw in enumerate(yaw_angles):
        for i in range(num_rp):
                    # Rotate Ht
                    Ht = item_heightmaps_RP[i,:,:,0]
                    rotated_Ht = rotate(Ht, yaw, reshape=False)
                    item_heightmaps_RPY[j,i,:,:,0] = rotated_Ht

                    # Rotate Hb
                    Hb = item_heightmaps_RP[i,:,:,1]
                    rotated_Hb = rotate(Hb, yaw, reshape=False)
                    item_heightmaps_RPY[j,i,:,:,1] = rotated_Hb 

    #-- conv_block visualize
    batch_size = 1
    conv_block_graph = draw_graph(conv_block(in_c=3,out_c=1), input_size=(batch_size,3,200,200),graph_name = 'conv_block',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/', graph_dir ='TB')
    conv_block_graph.visual_graph

    #-- Downsample visualize
    Downsample_graph = draw_graph(Downsample(channel=1), input_size=(batch_size,1,32,32),graph_name = 'Downsample',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/', graph_dir ='LR')
    Downsample_graph.visual_graph

    #-- Upsample visualize
    Upsample_graph = draw_graph(Upsample(channel=3), input_data=[torch.rand(batch_size,3,32,32), torch.rand(batch_size,3,64,64)],graph_name = 'Upsample',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/',  graph_dir ='LR')
    Upsample_graph.visual_graph""" 






