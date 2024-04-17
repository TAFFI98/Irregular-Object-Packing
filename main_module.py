import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
import torch
import pybullet as p
import pybullet_data

from trainer import Trainer
from tester import Tester
from env import Env

from logger import Logger
import utils

def main(args):

    # Setup environment
    is_sim = args.is_sim        # per adesso is_sim lo lascio fisso a True, poi da definire a livello di logica quando si implementa il robot fisico
    is_testing = args.is_testing
    obj_folder_path = args.obj_folder_path
    stage = args.stage

    # environment setup
    box_size=(0.4,0.4,0.3)
    resolution = 50
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=box_size, resolution = resolution)
    print('----------------------------------------')
    print('Setup of PyBullet environment: \nBox size: ',box_size, '\nResolution: ',resolution)

    # Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()
    print('----------------------------------------')
    print('Generating CSV with objects')

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
    K = args.k_obj
    item_numbers = np.random.randint(0, 100, size=K)
    item_ids = env.load_items(item_numbers)

    # volumes, sorted_ids = env.order_by_item_volume(item_ids)
    volume_bbox, bbox_order = env.order_by_bbox_volume(item_ids)

    #for i in range(500):
    #    p.stepSimulation()

    print('----------------------------------------')
    print('K = ', K, 'Items Loaded')
    print('--------------------------------------')
    print('Loaded ids before packing: ', env.loaded_ids)
    print('Packed ids before packing: ', env.packed)
    print('UnPacked ids before packing: ', env.unpacked)
    print('--------------------------------------')
    prev_obj = 0

    eps_height = 0.05 * box_size[2] # 5% of box height

    # loop over the loaded objects
    for i in range(K):
        item = item_ids[i]
        heightmap_box = env.box_heightmap()

        # check if item is already packed
        unpacked = env.unpacked

        if len(unpacked) == 0: 
            break

        # check if item is packable by maximum height
        max_Heightmap_box = np.max(heightmap_box)
        is_box_full = max_Heightmap_box > box_size[2] - eps_height
        
        if  is_box_full: 
            break
        
        # selection strategy according to the chosen stage

        if stage == 1:

            # pack largest object first
            print('---------------------------------------')
            print('Stage 1')
            
            next_obj = bbox_order[i]

        elif stage == 2:

            # select k objects with the largest boubding box volume
            print('---------------------------------------')
            print('Stage 2')
            
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
                    Ht,_,obj_length,obj_width  = env.item_hm(unpacked_ids[i], view)
                    #env.visualize_object_heightmaps(id_, view, only_top = True )
                    #env.visualize_object_heightmaps_3d(id_, view, only_top = True )
                    item_views.append(Ht)
                item_views = np.array(item_views)
                views.append(item_views)
            views = np.array(views)

            
    print('End of main_module.py')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser fro training')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store', default=True)
    parser.add_argument('--is_testing', dest='is_testing', action='store', default=False)
    parser.add_argument('--obj_folder_path', dest='obj_folder_path', action='store', default='objects/')
    parser.add_argument('--train', dest='train', action='store', default=False)
    parser.add_argument('--stage', dest='stage', action='store', default=2)
    parser.add_argument('--k_obj', dest='k_obj', action='store', default=5)
    parser.add_argument('--k_sort', dest='k_sort', action='store', default=2)

    args = parser.parse_args()
    main(args) 
