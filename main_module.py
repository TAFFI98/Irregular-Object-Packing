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

    #-- Compute bounding boxes  Volumes and orders them  - for stage 1
    volume_bbox, bbox_order = env.order_by_bbox_volume(env.loaded_ids)
    print(' The order by bbox volume is: ', bbox_order)

    print('End of main_module.py')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='simple parser fro training')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store', default=True)
    parser.add_argument('--is_testing', dest='is_testing', action='store', default=False)
    parser.add_argument('--obj_folder_path', dest='obj_folder_path', action='store', default='objects/')
    parser.add_argument('--train', dest='train', action='store', default=False)

    args = parser.parse_args()
    main(args)
