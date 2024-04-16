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
    K = 3
    item_numbers = np.random.randint(0, 100, size=K)
    item_ids = env.load_items(item_numbers)

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

    box_volume = box_size[0] * box_size[1] * box_size[2] # volume of the box, CHECK UNIT MEASURES!!!!!
    eps_volume = 0.1 * box_volume 

    total_volume = 0

    # loop over the loaded objects
    for i in range(K):
        item = item_ids[i]
        Hc = env.box_heightmap()

        # check if item is already packed
        unpacked = env.unpacked

        if len(unpacked) == 0 or is_box_full: 
            break

        # compute total volume with new item
        item_volume = env.item_volume(item)
        total_volume = total_volume + item_volume

        is_box_full = (box_volume - total_volume) < eps_volume # in altrenativa potrei controllare la heightmap, ma non sarebbe sufficiente calcolare il valore massimo sulla z.
        
        # check if item is packable by volume
        if  is_box_full: 
            break
        
        if stage == 1:

            # sort objects by volume 
            volumes, sorted_ids = env.order_by_item_volume(item_ids)


            print('Stage 1')

    print('End of main_module.py')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser fro training')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store', default=True)
    parser.add_argument('--is_testing', dest='is_testing', action='store', default=False)
    parser.add_argument('--obj_folder_path', dest='obj_folder_path', action='store', default='objects/')
    parser.add_argument('--train', dest='train', action='store', default=False)
    parser.add_argument('--stage', dest='stage', action='store', default=1)

    args = parser.parse_args()
    main(args) 
