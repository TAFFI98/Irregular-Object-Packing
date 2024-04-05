# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:34:07 2022

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import os
from env import Env

class HM_heuristic(object):
    def __init__(self, c = 0.01):
        self.c = c
        
    def pack(self, env, item):
        Hc = env.box_heightmap()
        if item in env.packed:
            print("item {} already packed!".format(item))
            return False
        pitch, roll = np.meshgrid(np.arange(0,2*np.pi,np.pi/2),np.arange(0,2*np.pi,np.pi/2))
        pitch_rolls = np.array([pitch.reshape(-1), roll.reshape(-1)]).T
        Trans = []
        BoxW, BoxH = env.resolution, env.resolution
        for pitch_roll in pitch_rolls:
            transforms = np.concatenate((np.repeat([pitch_roll],4,axis=0).T,[np.arange(0,2*np.pi,np.pi/2)]),axis=0).T
            for trans in transforms:       
                Ht, Hb = env.item_hm(item, trans)
                w,h = Ht.shape
                for X in range(0, BoxW-w+1):
                    for Y in range(0, BoxH-h+1):
                        Z = np.max(Hc[X:X+w, Y:Y+h]-Hb)
                        Update = np.maximum((Ht>0)*(Ht+Z), Hc[X:X+w,Y:Y+h])
                        if np.max(Update) <= env.box_size[2]:
                            score = self.c*(X+Y)+np.sum(Hc)+np.sum(Update)-np.sum(Hc[X:X+w,Y:Y+h])
                            Trans.append(np.array(list(trans)+[X,Y,Z,score]))
        if len(Trans)!=0:
            Trans = np.array(Trans)[np.argsort(np.array(Trans)[:,6])]
            trans = Trans[0]
        if type(trans)!=type(None):
            print("Pos:%d,%d,%f" % (trans[3],trans[4],trans[5]))
            S = env.pack_item(item, trans)
        return S

if __name__ == '__main__':

    #-- Path with the URDF files
    obj_folder_path = 'Irregular-Object-Packing/objects/'
    
    #-- PyBullet Environment setup 
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 50)

    #-- Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
    item_numbers = np.arange(80,95)
    item_ids = env.load_items(item_numbers)

    for i in range(500):
        p.stepSimulation()

    #-- Pack items using HM euristics
    HM = HM_heuristic()
    for i,item_ in enumerate(item_ids):
        stability = HM.pack(env,item_ )
        print(' Is the placement stable?', stability)
    print('Objects packed')
    

    