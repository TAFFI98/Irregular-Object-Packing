# -*- coding: utf-8 -*-
"""
Created on  Apr 3 2024

@author: Taffi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pybullet as p
import time
import pybullet_data
import os, csv

class Env(object):
    def __init__(self, obj_dir, is_GUI = True, box_size = (0.4,0.4,0.3), resolution = 40):
        
        self.p = p

        #-- Paths setup
        self.obj_dir = obj_dir                                              # directory with urdfs of objects
        self.csv_file = self.obj_dir + 'objects.csv'                        # path to the csv file with list of objects index,name, path
        
        #-- Box dimensions
        self.box_size = box_size                                            # size of the box
        length, width, height = box_size
        self.bbox_box = (0, 0, 0, length, width, height)                    # bounding box of the box

        wall_width = 0.1                                                    # width of the walls of the box
        boxL, boxW, boxH = box_size                                         # Box length width and height
        
        #-- Camera resolution
        self.resolution = resolution                                        # resolution of the camera     
        
        #-- Setup of PyBullet simulation
        if p.isConnected():
            p.disconnect()
        
        
        if is_GUI:
            self.physicsClient = p.connect(p.GUI)

        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf')

        #-- Create packing box in simulation
        wall_1_index = self.create_box([boxL,wall_width,boxH],[boxL/2,-wall_width/2,boxH/2])
        wall_2_index = self.create_box([boxL,wall_width,boxH],[boxL/2,boxW+wall_width/2,boxH/2])
        wall_3_index = self.create_box([wall_width,boxW,boxH],[-wall_width/2,boxW/2,boxH/2])
        wall_4_index = self.create_box([wall_width,boxW,boxH],[boxL+wall_width/2,boxW/2,boxH/2])
        self.walls_indices = [wall_1_index, wall_2_index, wall_3_index, wall_4_index]
        
        #-- Set mass to 0 to avoid box movements
        for i in range(4):
            p.changeDynamics(self.walls_indices[i], -1, mass=0) # Set mass to 0 to avoid box movements

        #-- Initialize ids of the objects loaded in simulation
        self.loaded_ids = []
        #-- Initialize ids of the objects already packed
        self.packed = []
        #-- Initialize ids of the objects not packed
        self.unpacked = []       
        
    def create_box(self, size, pos):
        '''
        size: list of 3 elements [length, width, height]
        pos: list of 3 elements [x, y, z]
        output: integer representing the index of the box in the simulation
        
        Function to create the walls of the packing box in simulation
        '''
        size = np.array(size)
        shift = [0, 0, 0]
        color = [1,1,1,1]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        rgbaColor=color,
                                        visualFramePosition=shift,
                                        halfExtents = size/2)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                              collisionFramePosition=shift,
                                              halfExtents = size/2)
        box_index = p.createMultiBody(baseMass=100,
                          baseInertialFramePosition=[0, 0, 0],
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=pos,
                          useMaximalCoordinates=True)
        return box_index
    
    def load_items(self, item_ids):
        '''
        item_ids: list of integers
        output: list of integers representind the loaded ids of the objects
        
        Function to load in simulation the objects on the basis of the ids
        '''
        flags = p.URDF_USE_INERTIA_FROM_FILE

        # Read URDF information from CSV file
        with open(self.csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            urdf_info = list(reader)

        for count, urdf_data in enumerate(urdf_info):
            if count in item_ids:
                urdf_path = urdf_data['URDF Path']
                print(urdf_path)
                loaded_id = p.loadURDF(urdf_path, [(count//5)/4+2.2, (count%5)/4+0.2, 0.1], flags=flags)
                self.loaded_ids.append(loaded_id)
                self.unpacked.append(loaded_id)
        return self.loaded_ids
    
    def remove_all_items(self):
        '''
        Function to remove from the simulation all the objects
        '''
        for loaded in self.loaded_ids:
            p.removeBody(loaded)
        self.loaded_ids = []
        
    def box_heightmap(self):
        '''
        output: numpy array of shape (resolution, resolution)

        Function to compute the box heigthmap
        '''
        sep = self.box_size[0]/self.resolution
        xpos = np.arange(sep/2,self.box_size[0]+sep/2,sep)
        ypos = np.arange(sep/2,self.box_size[1]+sep/2,sep)
        xscan, yscan = np.meshgrid(xpos,ypos)
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Start = np.insert(ScanArray,2,self.box_size[2],0).T
        End = np.insert(ScanArray,2,0,0).T
        RayScan = np.array(p.rayTestBatch(Start, End),dtype=object)
        Height = (1-RayScan[:,2].astype('float64'))*self.box_size[2]
        HeightMap = Height.reshape(self.resolution,self.resolution).T
        return HeightMap  
    
    def visualize_object_heightmaps(self, item_id, orient, only_top = False):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        only_top: boolean
        
        Function to visualize the object heightmaps (bottom and top) in 2D
        '''
        Ht,Hb, obj_length, obj_width = self.item_hm(item_id, orient)
        
        # Display the top-view and bottom-view heightmaps
        if only_top == False:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(Ht, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs[0].set_title('Top View Heightmap')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[1].imshow(Hb, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs[1].set_title('Bottom View Heightmap')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            plt.tight_layout()
            plt.show()
        if only_top == True:
            # Display the top-view  heightmap
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
            axs.imshow(Ht, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs.set_title('Top View Heightmap')
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            plt.tight_layout()
            plt.show()

    def visualize_object_heightmaps_3d(self, item_id, orient, only_top = False ):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        only_top: boolean
        
        Function to visualize the object heightmaps (bottom and top) in 3D
        '''
        Ht,Hb, obj_length, obj_width = self.item_hm(item_id, orient)

        # Create 3D grid
        X, Y = np.meshgrid(np.linspace(0, self.box_size[0], Ht.shape[1]),
                        np.linspace(0, self.box_size[1], Ht.shape[0]))
        if only_top == False:
            # Plot the top-view heightmap
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_surface(X, Y, Ht, cmap='viridis')
            ax1.set_title('Top View Heightmap')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Height')

            # Plot the bottom-view heightmap
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(X, Y, Hb, cmap='viridis')
            ax2.set_title('Bottom View Heightmap')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Height')
            plt.tight_layout()
            plt.show()

        if only_top == True:
            # Plot the box heightmap
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Ht, cmap='viridis')
            ax.set_title('Top Heightmap')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Height')
            plt.tight_layout()
            plt.show()

    def visualize_box_heightmap(self):
        '''
        Function to visualize the box heightmap in 2D
        '''
        self.Boxheightmap = self.box_heightmap()
        # Display the top-view  heightmap
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.imshow(self.Boxheightmap, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
        axs.set_title('Top View Heightmap')
        axs.set_xlabel('X')
        axs.set_ylabel('Y')
        plt.tight_layout()
        plt.show()
    
    def visualize_box_heightmap_3d(self):
        '''
        Function to visualize the box heightmap in 3D
        '''
        box_heightmap = self.box_heightmap()

        # Create 3D grid
        X, Y = np.meshgrid(np.linspace(0, self.box_size[0], box_heightmap.shape[1]),
                        np.linspace(0, self.box_size[1], box_heightmap.shape[0]))

        # Plot the box heightmap
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, box_heightmap, cmap='viridis')
        ax.set_title('Box Heightmap')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        
        plt.tight_layout()
        plt.show()

    def item_hm(self, item_id, orient):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        output: 2 numpy arrays of shape (resolution, resolution) - bottom and top heightmaps
        
        Function to compute the item (with a certain item_id) heigthmap
        '''
        
        # Stores the item original position and orientation
        old_pos, old_quater = p.getBasePositionAndOrientation(item_id)  
        quater = p.getQuaternionFromEuler(orient)
        
        # Set new orientation for the item
        p.resetBasePositionAndOrientation(item_id,[1,1,1],quater)       
        
        # Extracts the boundingbox of the item
        AABB = p.getAABB(item_id)                                      
        obj_length = AABB[1][0] - AABB[0][0]
        obj_width = AABB[1][1] - AABB[0][1]
         # Computation of a grid of points within the specified box size
        sep = self.box_size[0] / self.resolution
        xpos = np.arange(sep / 2, self.box_size[0] + sep / 2, sep)
        ypos = np.arange(sep / 2, self.box_size[1] + sep / 2, sep)
        xscan, yscan = np.meshgrid(xpos, ypos)
        
        # Adjust the position of the grid to center the object in the 2D image
        x_offset = (self.box_size[0] - obj_length) / 2
        y_offset = (self.box_size[1] - obj_width) / 2
        xscan += AABB[0][0] - x_offset
        yscan += AABB[0][1] - y_offset
        

        # Ray casting from the top to the bottom and from the bottom to the top of the item's bounding box 
        # to find the height of points on the surface of the bounding box.
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Top = np.insert(ScanArray,2,AABB[1][2],axis=0).T
        Down = np.insert(ScanArray,2,AABB[0][2],axis=0).T
        RayScanTD = np.array(p.rayTestBatch(Top, Down),dtype=object)
        RayScanDT = np.array(p.rayTestBatch(Down, Top),dtype=object)
        
        # Computes the heights of points on the top (Ht) and bottom (Hb) surfaces of the bounding box based on the results of the ray casting.
        Ht = (1-RayScanTD[:,2])*(AABB[1][2]-AABB[0][2])
        RayScanDT = RayScanDT[:,2]
        RayScanDT[RayScanDT==1] = np.inf
        Hb = RayScanDT*(AABB[1][2]-AABB[0][2])
        # Replace infinity values in Hb with the maximum height of the bounding box
        max_height = AABB[1][2] - AABB[0][2]
        Hb[Hb == np.inf] = 0
        Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
        Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
        
        # Resets the initial orientation for the item
        p.resetBasePositionAndOrientation(item_id,old_pos,old_quater)
        return Ht,Hb, obj_length, obj_width
    


    
    def order_by_bbox_volume(self,items_ids):
        '''
        items_ids: list of integers
        output: list of floats (bounding boxes volumes), list of integers (ordered ids)
        
        Function to determine the order of objects based on their bounding box volumes in descending order.
        '''
        volume = []
        for item in items_ids:
            AABB = np.array(p.getAABB(item))
            volume.append(np.product(AABB[1]-AABB[0]))
        bbox_order = np.argsort(volume)[::-1]
        pybullet_ordered_ids = np.asarray(items_ids)[bbox_order]
        return volume, pybullet_ordered_ids
    
    def compute_object_bbox(self,item):
        '''
        item: integer representing the id of the object
        output: numpy array of shape (2,3) representing the bounding box of the object
        
        Function to compute the bounding box of an item
        '''
        AABB = np.array(p.getAABB(item))
        return AABB
    
    def order_by_item_volume(self,items_ids):
        '''
        items_ids: list of integers representing the ids of the items
        output: list of floats (objects volumes), list of integers (ordered ids)
        
        Function to determine the order of otems based on their volumes in descending order.
        '''
        volume = []
        for item in items_ids:
            volume.append(self.item_volume(item))
        vol_order = np.argsort(volume)[::-1]
        pybullet_ordered_ids = np.asarray(items_ids)[vol_order]
        return volume, pybullet_ordered_ids

    def grid_scan(self, xminmax, yminmax, z_start, z_end, sep):
        '''
        xminmax: list of 2 elements [xmin, xmax] representing the x-axis limits of the scanning operation
        yminmax: list of 2 elements [ymin, ymax] representing the y-axis limits of the scanning operation
        z_start: float representing the start height of the scanning operation
        z_end: float representing the end height of the scanning operation
        sep: float representing the separation between the points in the grid
        output: numpy array of shape (resolution, resolution) representing the heightmap


        This function performs a scanning operation in a grid pattern within a specified 3D space
        '''

        xpos = np.arange(xminmax[0]+sep/2,xminmax[1]+sep/2,sep)
        ypos = np.arange(yminmax[0]+sep/2,yminmax[1]+sep/2,sep)
        xscan, yscan = np.meshgrid(xpos, ypos)
        ScanArray = np.array([xscan.reshape(-1), yscan.reshape(-1)])
        Start = np.insert(ScanArray, 2, z_start,0).T
        End = np.insert(ScanArray, 2, z_end, 0).T
        RayScan = np.array(p.rayTestBatch(Start, End),dtype=object)
        Height = RayScan[:,2].astype('float64')*(z_end-z_start)+z_start
        HeightMap = Height.reshape(ypos.shape[0],xpos.shape[0]).T
        return HeightMap
    
    def item_volume(self, item):
        '''
        item: integer representing the id of the object
        output: float representing the volume of the object
        
        Function to compute the volume of an item, taking into account rotations and scans to find the tightest enclosing volume.
        ''' 
        scan_sep = 0.005
        old_pos, old_quater = p.getBasePositionAndOrientation(item)
        volume = np.inf # a big number
        for row in np.arange(0, 2*np.pi, np.pi/4):
            for pitch in np.arange(0, 2*np.pi, np.pi/4):
                quater = p.getQuaternionFromEuler([row, pitch, 0])
                p.resetBasePositionAndOrientation(item,[1,1,1],quater)
                AABB = p.getAABB(item)
                TopDown = self.grid_scan([AABB[0][0], AABB[1][0]], [AABB[0][1],AABB[1][1]],
                                    AABB[1][2], AABB[0][2], scan_sep)
                DownTop = self.grid_scan([AABB[0][0], AABB[1][0]], [AABB[0][1],AABB[1][1]],
                                    AABB[0][2], AABB[1][2], scan_sep)
                HeightDiff = TopDown-DownTop
                HeightDiff[HeightDiff<0] = 0 # empty part
                temp_v = np.sum(HeightDiff)*(scan_sep/0.01)**2
                volume = min(volume, temp_v)
        p.resetBasePositionAndOrientation(item,old_pos,old_quater)
        return volume
    
    def pack_item(self, item_id, transform):
        '''
        item_id: integer representing the id of the object
        transform: list of 6 elements [target_euler, target_pos]
        outputs:
        NewBoxHeightMap: numpy array of shape (resolution, resolution) representing the box heightmap after the packing operation
        stability: integer representing the stability of the packing
        old_pos: list of 3 elements representing the old position of the item
        old_quater: list of 4 elements representing the old orientation of the item

        Function to reset the position and orientation of the item based on the provided transform argument[target_euler,target_pos]. It updates the environment attributes (env.unpacked and env.packed) accordingly.
        '''
        old_pos, old_quater = p.getBasePositionAndOrientation(item_id) 
        if item_id in self.packed:
            print("item {} already packed!".format(item_id))
            return False
        z_shift = 0.005
        target_euler = transform[0:3]
        target_pos = transform[3:6]
        pos, quater = p.getBasePositionAndOrientation(item_id)
        new_quater = p.getQuaternionFromEuler(target_euler)
        p.resetBasePositionAndOrientation(item_id, pos, new_quater)
        AABB = p.getAABB(item_id)
        shift = np.array(pos)-(np.array(AABB[0])+np.array([self.box_size[0]/2/self.resolution,
                         self.box_size[1]/2/self.resolution,z_shift]))
        new_pos = target_pos+shift
        p.resetBasePositionAndOrientation(item_id, new_pos, new_quater)
        for i in range(500):
            p.stepSimulation()
            time.sleep(1./240.)

        #-- After the simulation, it checks the stability of the item based on its final position and orientation.
        curr_pos, curr_quater = p.getBasePositionAndOrientation(item_id)
        curr_euler = np.array(p.getEulerFromQuaternion(curr_quater))
        #-- Stability (boolean) is determined by checking if the item has settled within a small tolerance of its target position and orientation. 
        stability_bool = np.linalg.norm(new_pos-curr_pos)<0.02 and curr_euler.dot(target_euler)/(np.linalg.norm(curr_euler)*np.linalg.norm(target_euler)) > np.pi*2/3
        stability = 1 if stability_bool else 0        
        self.packed.append(item_id)
        self.unpacked.remove(item_id)
        NewBoxHeightMap = self.box_heightmap()
        return NewBoxHeightMap, stability, old_pos, old_quater
    
    def unpack_item(self, item_id, old_pos, old_quater):
        '''
        item_id: integer representing the id of the object
        old_pos: list of 3 elements representing the old position of the item
        old_quater: list of 4 elements representing the old orientation of the item

        Restores the position and orientation of the item to its original state and updates env (env.packed and env.packed) attributes accordingly.
        '''
        p.resetBasePositionAndOrientation(item_id,old_pos,old_quater)
        self.packed.remove(item_id)
        self.unpacked.append(item_id)
         
    def drawAABB(self, aabb, width=1):
        '''
        aabb: list of 2 lists representing the bounding box of the object
        width: integer representing the width of the lines
        output: list of integers representing the line ids

        Function to draw a BBox in simulation
        '''
        aabbMin = aabb[0]
        aabbMax = aabb[1]
        line_ids = [] # List to store the line ids
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMin[2]]
        line_id = p.addUserDebugLine(f, t, [1, 0, 0], width)
        line_ids.append(line_id)
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        line_id = p.addUserDebugLine(f, t, [0, 1, 0], width)
        line_ids.append(line_id)
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMin[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [0, 0, 1], width)
        line_ids.append(line_id)
        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMin[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        line_id = p.addUserDebugLine(f, t, [1, 1, 1], width)
        line_ids.append(line_id)
        return line_ids

    def removeAABB(self, line_ids):
        '''
        line_ids: list of line ids returned by drawAABB()

        Function to remove a BBox from simulation
        '''
        for line_id in line_ids:
            p.removeUserDebugItem(line_id)

    def draw_box(self, width=5):
        '''
        width: integer representing the width of the lines

        Function to draw the Bounding Box of the package in simulation
        '''
        xmax = self.box_size[0]
        ymax = self.box_size[1]
        p.addUserDebugLine([0,0,0],[0,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,0],[0,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,0],[xmax,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,ymax,0],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,0,0],[xmax,0,0], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,0],[xmax,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([0,0,0],[0,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,0],[xmax,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([0,0,self.box_size[2]],[xmax,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,self.box_size[2]],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,0,self.box_size[2]],[0,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,self.box_size[2]],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)
    
    def generate_urdf_csv(self):
        '''
        output: integer representing the total number of objects

        Function that creates the CSV file with the object number, name and path
        '''
        with open(self.csv_file, 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Object Name', 'URDF Path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            # Initialize index
            index = 0

            # Recursively traverse folders
            for root, dirs, files in os.walk(self.obj_dir):
                for filename in sorted(files):
                    if filename.endswith('.urdf'):
                        # Extract object name from parent folder
                        object_name = os.path.basename(root)                        
                        # Construct URDF path
                        urdf_path = os.path.join(root, filename)
                        
                        # Write row to CSV
                        writer.writerow({'Index': index, 'Object Name': object_name, 'URDF Path': urdf_path})
                        index += 1
            tot_num_objects = index
            print('Total number of available objects: ', tot_num_objects )

        return tot_num_objects
    
    def Compactness(self, item_in_box, item_volumes, box_hm):
        '''
        item_in_box: list of integers representing the ids of the items in the box
        item_volumes: list of floats representing the volumes of the items
        box_hm: numpy array of shape (resolution, resolution) representing the box heightmap
        output: float representing the compactness of the packing

        Function to compute the compactness of the packing
        '''
        total_volume = 0
        for i,_ in enumerate(item_in_box):
            total_volume += item_volumes[i]
        box_volume = np.max(box_hm)*self.box_size[0]*self.box_size[1]
        print('Compactness is: ', total_volume/box_volume)
        return total_volume/box_volume

    def Pyramidality(self, item_in_box, item_volumes, box_hm):
        '''
        item_in_box: list of integers representing the ids of the items in the box
        item_volumes: list of floats representing the volumes of the items
        box_hm: numpy array of shape (resolution, resolution) representing the box heightmap
        output: float representing the piramidality of the packing
        
        Function to compute the piramidality of the packing
        '''
        total_volume = 0
        for i,item in enumerate(item_in_box):
            total_volume += item_volumes[i]
        used_volume = np.sum(box_hm)
        print('Piramidality is: ', total_volume/used_volume)
        return total_volume/used_volume
    
    def Objective_function(self, item_in_box, item_volumes, box_hm, stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25):
        '''
        item_in_box: list of integers representing the ids of the items in the box
        item_volumes: list of floats representing the volumes of the items
        box_hm: numpy array of shape (resolution, resolution) representing the box heightmap
        stability_of_packing: float representing the stability of the packing
        alpha: float representing the weight of the compactness term in the objective function
        beta: float representing the weight of the pyramidality term in the objective function
        gamma: float representing the weight of the stability term in the objective function
        output: float representing the objective function value
        
        Function to compute the objective function value based on the compactness, pyramidality, and stability of the packing
        '''
        obj  = alpha * self.Compactness(item_in_box, item_volumes, box_hm) + beta * self.Pyramidality(item_in_box, item_volumes, box_hm) + gamma * stability_of_packing
        print('Objective function is: ', obj)
        return obj
    
    def Reward_function(self, obj_1, obj_2):
        '''
        obj_1: float representing the objective function value before the action
        obj_2: float representing the objective function value after the action
        output: float representing the reward
        
        Function to compute the reward based on the difference between the objective function values before and after the action
        '''
        reward  = obj_2 - obj_1
        print('------------------Reward  is: ', reward,'  ------------------')
        return reward
    
    def get_z(self,Hc,Hb,pixel_x,pixel_y,obj_length,obj_width):
        '''
        Hc: numpy array of shape (resolution, resolution) representing the current box heightmap
        Hb: numpy array of shape (resolution, resolution) representing the heightmap of the item
        pixel_x: integer representing the x-coordinate of the pixel representing the item position in the box
        pixel_y: integer representing the y-coordinate of the pixel representing the item position in the box
        output: float representing the maximum z-value of the item in the box
        
        Function to compute the maximum z-value of the item in the box due to gravity
        '''

        s_range = range(-Hb.shape[0]//2, Hb.shape[0]//2)
        t_range = range(-Hb.shape[1]//2, Hb.shape[1]//2)
        max_z = 0
        ss_range = range(-(int(self.resolution*obj_length//self.box_size[0]))//2, int((self.resolution*obj_length//self.box_size[0]))//2)
        tt_range = range(-(int(self.resolution*obj_width//self.box_size[1]))//2, int((self.resolution*obj_width//self.box_size[1]))//2)

        if np.all(Hc == 0): # No items in the box
            max_z = 0
        else:
            for s in s_range:
                for t in t_range:
                        for ss in ss_range:
                            for tt in tt_range:
                                if 0 <= pixel_x+ss < Hc.shape[0] and 0 <= pixel_y+tt < Hc.shape[1]:
                                    z = Hc[pixel_x+ss, pixel_y+tt] - Hb[s, t]
                                    max_z = max(max_z, z)
        return max_z

    def is_box_full(heightmap, max_height):
        '''
        heightmap: numpy array of shape (resolution, resolution) representing the box heightmap
        max_height: float representing the maximum height of the box
        output: boolean
        
        Function to check if the box is full
        '''
        return np.all(heightmap == max_height)
    
if __name__ == '__main__':

 #-- Path with the URDF files
    obj_folder_path = 'objects/'
    
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


    #-- Compute Box HeightMap: shape (resolution, resolution)
    BoxHeightMap = env.box_heightmap()
    print('----------------------------------------')
    print('Computed Box Height Map')
    for i in range(500):
        p.stepSimulation()


    #-- Compute bounding boxes Items Volumes and orders them  
    volume_bbox, bbox_order = env.order_by_bbox_volume(env.loaded_ids)
    print(' The order by bbox volume is: ', bbox_order)
        

    #-- Compute  Items Volumes and orders them  
    volume_items, volume_order = env.order_by_item_volume(env.loaded_ids)
    print(' The order by item volume is: ', volume_order)


    #-- Pack items with random transformation
    prev_obj = 0
    for i,item_ in enumerate(bbox_order):
        print('Packing item ', i+1, 'with id: ', item_)
        target_euler = [0,0,0]
        target_pos = [20-i*5,20-i*5,0] # cm
        transform = np.empty(6,)
        transform[0:3] = target_euler
        transform[3:6] = target_pos
        ''' Pack item '''
        NewBoxHeightMap, stability_of_packing = env.pack_item(item_ , transform)
        print(' Is the placement stable?', stability_of_packing)
        volume_items_packed, _ = env.order_by_item_volume(env.packed)
        current_obj = env.Objective_function(env.packed, volume_items_packed, NewBoxHeightMap, stability_of_packing, alpha = 0.75, beta = 0.25, gamma = 0.25)
        if i>= 1:
            ''' Compute reward '''
            Reward = env.Reward_function(prev_obj, current_obj)
        prev_obj = current_obj

    env.visualize_box_heightmap()
    env.visualize_box_heightmap_3d()
    
    for i in range(500):
        p.stepSimulation()

    #-- Remove all items 
    item_ids = env.remove_all_items()

    for i in range(500):
        p.stepSimulation()