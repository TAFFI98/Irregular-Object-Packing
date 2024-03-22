from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')
from models import conv_block, Upsample, Downsample, final_conv_select_net, placement_net, feat_extraction_select_net
import torch
import pybullet as p
from env import Env
import numpy as np
from scipy.ndimage import rotate

if __name__ == '__main__':
        

    #-- Path with the URDF files
    obj_folder_path = '/Project/Irregular-Object-Packing/objects/'
    
    #-- PyBullet Environment setup 
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 50)

    #-- Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
    item_numbers = np.arange(84,86)
    item_ids = env.load_items(item_numbers)

    for i in range(500):
        p.stepSimulation()

    #-- Compute Box HeightMap: shape (resolution, resolution)
    BoxHeightMap = env.box_heightmap()
    print(' The box heightmap is: ', BoxHeightMap)

    for i in range(500):
        p.stepSimulation()

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
    for id_ in item_ids:
        six_views_one_item = []
        for view in principal_views.values():
            Ht,_ = env.item_hm(id_, view )
            env.visualize_object_heightmaps(id_, view, only_top = True )
            env.visualize_object_heightmaps_3d(id_, view, only_top = True )
            six_views_one_item.append(Ht) 
        six_views_one_item = np.array(six_views_one_item)
        six_views_all_items.append(six_views_one_item)
    six_views_all_items = np.array(six_views_all_items) # (n_loaded_items, 6, resolution, resolution)

    # -- Visualize selection network - I choose the first item
    six_views_chosen_item = six_views_all_items[0,:,:,:] # (6, resolution, resolution)
    
    # -- I concatenate the box heightmap    
    input_selection_network = np.concatenate((np.expand_dims(BoxHeightMap, axis=0), six_views_chosen_item), axis=0) # (7, resolution, resolution)
   
    # to account for batch = 1
    input_selection_network = np.expand_dims(input_selection_network, axis=0) 
    
    #-- feat_extraction_select_net visulization 
    feat_extraction_select_net = draw_graph(feat_extraction_select_net(use_cuda=False), input_data=[torch.tensor(input_selection_network.astype('float32'))],graph_name = 'feat_extraction_select_net',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/')
    feat_extraction_select_net.visual_graph

    #-- final_conv_select_net visulization 
    batch_size = 30
    K = 10
    final_conv_select_net_graph = draw_graph(final_conv_select_net(use_cuda=False,K = K), input_size=(batch_size,512,K),graph_name = 'final_conv_select_net',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/')
    final_conv_select_net_graph.visual_graph


    #-- Choose one Item and compute HeightMaps varying first roll, pitch and then yaw (according to the paper)
    item_id = item_ids[0]
    
    #-- Discretize roll, pitch and yaw
    roll_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    pitch_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    num_roll, num_pitch = roll_angles.shape[0], pitch_angles.shape[0]
    yaw_angles = np.arange(0,2*np.pi,np.pi/2)*180/np.pi
    num_yaws = yaw_angles.shape[0]

    item_heightmaps_RP = []
    roll_pitch_angles = []

    for r,roll in enumerate(roll_angles):
            for p,pitch in enumerate(pitch_angles):
                orient = [roll, pitch, 0]
                roll_pitch_angles.append(np.array([roll,pitch]))
                Ht,Hb = env.item_hm(item_id, orient )
                env.visualize_object_heightmaps(item_id, orient)
                env.visualize_object_heightmaps_3d(item_id,orient )
                Ht.shape = (Ht.shape[0], Ht.shape[1], 1)
                Hb.shape = (Hb.shape[0], Hb.shape[1], 1)
                item_heightmap = np.concatenate((Ht, Hb), axis=2)
                item_heightmaps_RP.append(item_heightmap)  

    item_heightmaps_RP = np.asarray(item_heightmaps_RP)  

    num_rp = item_heightmaps_RP.shape[0]
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

    # Selection of j = 1 (determines yaw) and i = 3 (determines roll and pitch) to test the networks  - from item_heightmaps_RPY of shape: (num_yaws, num_RP, resolution, resolution, 2) -
    
    input_placement_network = np.transpose(np.concatenate((np.expand_dims(BoxHeightMap, axis=2), item_heightmaps_RPY[1,3,:,:]), axis=2), (2, 0, 1))
   
    # to accounto for batch = 1
    input_placement_network = np.expand_dims(input_placement_network, axis=0) 
    
    #-- placement_net visulize
    placement_net_graph = draw_graph(placement_net(), input_data=[torch.tensor(input_placement_network.astype('float32'))],graph_name = 'placement_net',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/')
    placement_net_graph.visual_graph

    #- Visualize BoxHeightMap, Ht, Bb for the chosen orientation
    yaw = yaw_angles[1]
    roll = roll_pitch_angles[3][0]
    pitch = roll_pitch_angles[3][1]
    orient = [roll,pitch,yaw]
    env.visualize_object_heightmaps(item_id, orient)
    env.visualize_object_heightmaps_3d(item_id,orient )


    #-- conv_block visualize
    batch_size = 1
    conv_block_graph = draw_graph(conv_block(in_c=3,out_c=1), input_size=(batch_size,3,200,200),graph_name = 'conv_block',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/', graph_dir ='TB')
    conv_block_graph.visual_graph

    #-- Downsample visualize
    Downsample_graph = draw_graph(Downsample(channel=1), input_size=(batch_size,1,32,32),graph_name = 'Downsample',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/', graph_dir ='LR')
    Downsample_graph.visual_graph

    #-- Upsample visualize
    Upsample_graph = draw_graph(Upsample(channel=3), input_data=[torch.rand(batch_size,3,32,32), torch.rand(batch_size,3,64,64)],graph_name = 'Upsample',save_graph= True, directory= '/Project/Irregular-Object-Packing/models_plot/',  graph_dir ='LR')
    Upsample_graph.visual_graph





