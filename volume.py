# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 01:10:07 2022

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import os
#from pybullet_object_models import ycb_objects

xmin, xmax, ymin, ymax = 0.,0.4,0.,0.4
resolution = 40
TopHeight = 0.3
N = 20
#item_size = np.array([[5,5,8],[3,5,2],[1,3,4],[2,2,4],[6,2,4],[5,9,3],[6,8,1]])
item_size = np.random.randint(low=1,high=10,size=[N,3])
items = []
for size in item_size:
    items.append(np.ones(size))
c = 0.001

def load_items(numbers):
    flags = p.URDF_USE_INERTIA_FROM_FILE
    model_list = []
    item_ids = []
    for root,dirs,files in os.walk(r'D:\packing\pybullet-URDF-models-main\pybullet-URDF-models\urdf_models\models'):
        for file in files:
            if file == "model.urdf":
                model_list.append(os.path.join(root,file))
    for count in range(len(numbers)):
        item_id = p.loadURDF(model_list[numbers[count]-1], 
                             [(count//8)/3+1.2, (count%8)/3+0.2, 0.05], flags=flags)
        item_ids.append(item_id)
    return item_ids

def grid_scan(xminmax, yminmax, z_start, z_end, sep):
    xpos = np.arange(xminmax[0]+sep/2,xminmax[1]+sep/2,sep)
    ypos = np.arange(yminmax[0]+sep/2,yminmax[1]+sep/2,sep)
    xscan, yscan = np.meshgrid(xpos, ypos)
    ScanArray = np.array([xscan.reshape(-1), yscan.reshape(-1)])
    Start = np.insert(ScanArray, 2, z_start,0).T
    End = np.insert(ScanArray, 2, z_end, 0).T
    RayScan = np.array(p.rayTestBatch(Start, End))
    Height = RayScan[:,2].astype('float64')*(z_end-z_start)+z_start
    HeightMap = Height.reshape(ypos.shape[0],xpos.shape[0]).T
    return HeightMap

def box_hm():
    sep = (xmax-xmin)/resolution
    xpos = np.arange(xmin+sep/2,xmax+sep/2,sep)
    ypos = np.arange(ymin+sep/2,ymax+sep/2,sep)
    xscan, yscan = np.meshgrid(xpos,ypos)
    ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
    Start = np.insert(ScanArray,2,TopHeight,0).T
    End = np.insert(ScanArray,2,0,0).T
    RayScan = np.array(p.rayTestBatch(Start, End))
    Height = (1-RayScan[:,2].astype('float64'))*TopHeight
    HeightMap = Height.reshape(resolution,resolution).T
    return HeightMap  

def item_volume(item):
    # cm^3 
    scan_sep = 0.005
    old_pos, old_quater = p.getBasePositionAndOrientation(item)
    volume = np.inf # a big number
    for row in np.arange(0, 2*np.pi, np.pi/4):
        for pitch in np.arange(0, 2*np.pi, np.pi/4):
            quater = p.getQuaternionFromEuler([row, pitch, 0])
            p.resetBasePositionAndOrientation(item,[1,1,1],quater)
            AABB = p.getAABB(item)
            TopDown = grid_scan([AABB[0][0], AABB[1][0]], [AABB[0][1],AABB[1][1]],
                                AABB[1][2], AABB[0][2], scan_sep)
            DownTop = grid_scan([AABB[0][0], AABB[1][0]], [AABB[0][1],AABB[1][1]],
                                AABB[0][2], AABB[1][2], scan_sep)
            HeightDiff = TopDown-DownTop
            HeightDiff[HeightDiff<0] = 0 # empty part
            temp_v = np.sum(HeightDiff)*(scan_sep/0.01)**2
            volume = min(volume, temp_v)
    p.resetBasePositionAndOrientation(item,old_pos,old_quater)
    return volume

def Compactness(item_in_box, item_volumes, box_hm):
    total_volume = 0
    for item in item_in_box:
        total_volume += item_volumes[item]
    box_volume = np.max(box_hm)*box_hm.size
    return total_volume/box_volume

def Pyramidality(item_in_box, item_volumes, box_hm):
    total_volume = 0
    for item in item_in_box:
        total_volume += item_volumes[item]
    used_volume = np.sum(box_hm)
    return total_volume/used_volume

def item_hm(item,orient):
    old_pos, old_quater = p.getBasePositionAndOrientation(item)
    quater = p.getQuaternionFromEuler(orient)
    p.resetBasePositionAndOrientation(item,[1,1,1],quater)
    AABB = p.getAABB(item)
    sep = (xmax-xmin)/resolution
    xpos = np.arange(AABB[0][0]+sep/2,AABB[1][0],sep)
    ypos = np.arange(AABB[0][1]+sep/2,AABB[1][1],sep)
    xscan, yscan = np.meshgrid(xpos,ypos)
    ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
    Top = np.insert(ScanArray,2,AABB[1][2],axis=0).T
    Down = np.insert(ScanArray,2,AABB[0][2],axis=0).T
    RayScanTD = np.array(p.rayTestBatch(Top, Down))
    RayScanDT = np.array(p.rayTestBatch(Down, Top))
    Ht = (1-RayScanTD[:,2])*(AABB[1][2]-AABB[0][2])
    RayScanDT = RayScanDT[:,2]
    RayScanDT[RayScanDT==1] = np.inf
    Hb = RayScanDT*(AABB[1][2]-AABB[0][2])
    Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
    Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
    p.resetBasePositionAndOrientation(item,old_pos,old_quater)
    return Ht,Hb

def drawAABB(aabb,width=1):
  aabbMin = aabb[0]
  aabbMax = aabb[1]
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 0, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [0, 1, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [0, 0, 1], width)

  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMin[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)


if __name__ == '__main__':
    if p.getConnectionInfo()['isConnected']:
        p.disconnect()
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    item_numbers = np.arange(20,21)
    item_ids = load_items(item_numbers)
    v = []
    for item in item_ids:
        AABB = p.getAABB(item)
        drawAABB(AABB)
        volume = item_volume(item)
        v.append(volume)
        
        
    #pause
    '''
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)
        '''