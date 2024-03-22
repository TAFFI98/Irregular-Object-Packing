# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:53:21 2021

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pybullet as p
import time
import pybullet_data
import os, csv

class Env(object):
    def __init__(self, obj_dir, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 40):
        
        #-- Paths setup
        self.obj_dir = obj_dir                                  # directory with urdfs of objects
        self.csv_file = self.obj_dir + 'objects.csv'            # path to the csv file with list of objects index,name, path
        
        #-- Box dimensions
        self.box_size = box_size                                # size of the box
        wall_width = 0.1                                        # width of the walls of the box
        boxL, boxW, boxH = box_size                             # Box length width and height
        
        #-- Camera resolution
        self.resolution = resolution                            # resolution of the camera     
        
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
        self.create_box([boxL,wall_width,boxH],[boxL/2,-wall_width/2,boxH/2])
        self.create_box([boxL,wall_width,boxH],[boxL/2,boxW+wall_width/2,boxH/2])
        self.create_box([wall_width,boxW,boxH],[-wall_width/2,boxW/2,boxH/2])
        self.create_box([wall_width,boxW,boxH],[boxL+wall_width/2,boxW/2,boxH/2])
        
        #-- Initialize ids of the objects loaded in simulation
        self.loaded_ids = []
        #-- Initialize ids of the objects already packed
        self.packed = []
        #-- Initialize ids of the objects not packed
        self.unpacked = []       
        
    def create_box(self, size, pos):
        #-- Function to create the walls of the packing box
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
        p.createMultiBody(baseMass=100,
                          baseInertialFramePosition=[0, 0, 0],
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=pos,
                          useMaximalCoordinates=True)
        
    def load_items(self, item_ids):
        #-- Function to load in simulation the objects on the basis of the ids
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
        #-- Function to remove from the simulation all the objects 

        for loaded in self.loaded_ids:
            p.removeBody(loaded)
        self.loaded_ids = []
        
    def box_heightmap(self):
        #-- Function to compute the box heigthmap 
        sep = self.box_size[0]/self.resolution
        xpos = np.arange(sep/2,self.box_size[0]+sep/2,sep)
        ypos = np.arange(sep/2,self.box_size[1]+sep/2,sep)
        xscan, yscan = np.meshgrid(xpos,ypos)
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Start = np.insert(ScanArray,2,self.box_size[2],0).T
        End = np.insert(ScanArray,2,0,0).T
        RayScan = np.array(p.rayTestBatch(Start, End))
        Height = (1-RayScan[:,2].astype('float64'))*self.box_size[2]
        HeightMap = Height.reshape(self.resolution,self.resolution).T
        return HeightMap  
    
    def visualize_object_heightmaps(self, item_id, orient, only_top = False):
        # Visualize the object heightmaps in 2D
        Ht, Hb = self.item_hm(item_id, orient)
        
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

    def visualize_object_heightmaps_3d(self, item_id, orient, only_top = False  ):
        # Visulize the object heightmaps in 3d
        Ht, Hb = self.item_hm(item_id, orient)

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
        # Visualize the object heightmaps in 2D
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
        # Compute box heightmap
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
        #-- Function to compute the item (with a certain item_id) heigthmap 
        
        # Stores the item original position and orientation
        old_pos, old_quater = p.getBasePositionAndOrientation(item_id)  
        quater = p.getQuaternionFromEuler(orient)
        
        # Set new orientation for the item
        p.resetBasePositionAndOrientation(item_id,[1,1,1],quater)       
        
        # Extracts the boundingbox of the item
        AABB = p.getAABB(item_id)                                      
        
        # Computation of a grid of points within the bounding box of the item. 
        sep_x = (AABB[1][0] - AABB[0][0])/self.resolution
        sep_y = (AABB[1][1] - AABB[0][1])/self.resolution
        xpos = np.arange(AABB[0][0]+sep_x/2,AABB[1][0],sep_x)
        ypos = np.arange(AABB[0][1]+sep_y/2,AABB[1][1],sep_y)
        xscan, yscan = np.meshgrid(xpos,ypos)
        

        # Ray casting from the top to the bottom and from the bottom to the top of the item's bounding box 
        # to find the height of points on the surface of the bounding box.
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Top = np.insert(ScanArray,2,AABB[1][2],axis=0).T
        Down = np.insert(ScanArray,2,AABB[0][2],axis=0).T
        RayScanTD = np.array(p.rayTestBatch(Top, Down))
        RayScanDT = np.array(p.rayTestBatch(Down, Top))
        
        # Computes the heights of points on the top (Ht) and bottom (Hb) surfaces of the bounding box based on the results of the ray casting.
        Ht = (1-RayScanTD[:,2])*(AABB[1][2]-AABB[0][2])
        RayScanDT = RayScanDT[:,2]
        RayScanDT[RayScanDT==1] = np.inf
        Hb = RayScanDT*(AABB[1][2]-AABB[0][2])
        Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
        Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
        
        # Resets the initial orientation for the item
        p.resetBasePositionAndOrientation(item_id,old_pos,old_quater)
        return Ht,Hb
    
    def order_by_bbox_volume(self,items_ids):
        #-- Function to determine the order of bounding boxes based on their volumes in descending order.
        volume = []
        for item in items_ids:
            AABB = np.array(p.getAABB(item))
            volume.append(np.product(AABB[1]-AABB[0]))
        bbox_order = np.argsort(volume)[::-1]
        return volume, bbox_order
    
    def order_by_item_volume(self,items_ids):
        #-- Function to determine the order of bounding boxes based on their volumes in descending order.
        volume = []
        for item in items_ids:
            volume.append(self.item_volume(item))
        vol_order = np.argsort(volume)[::-1]
        return volume, vol_order

    def grid_scan(self, xminmax, yminmax, z_start, z_end, sep):
        
        # This function performs a scanning operation in a grid pattern within a specified 3D space
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
    
    def item_volume(self, item):
        # cm^3 
        # calculating the volume of an item, taking into account rotations and scans to find the tightest enclosing volume.
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
        #-- Function to reset the position and orientation of the item based on the provided transform argument[target_euler,target_pos]. 
        #-- target_pos in cm
        if item_id in self.packed:
            print("item {} already packed!".format(item_id))
            return False
        z_shift = 0.005
        target_euler = transform[0:3]
        target_pos = transform[3:6]
        target_pos[0] = target_pos[0]/self.resolution*self.box_size[0]
        target_pos[1] = target_pos[1]/self.resolution*self.box_size[1]
        pos, quater = p.getBasePositionAndOrientation(item_id)
        new_quater = p.getQuaternionFromEuler(target_euler)
        p.resetBasePositionAndOrientation(item_id, pos, new_quater)
        AABB = p.getAABB(item_id)
        shift = np.array(pos)-(np.array(AABB[0])+np.array([self.box_size[0]/2/self.resolution,
                         self.box_size[1]/2/self.resolution,z_shift]))
        new_pos = target_pos+shift
        p.resetBasePositionAndOrientation(item_id, new_pos, new_quater)
        for i in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        #-- After the simulation, it checks the stability of the item based on its final position and orientation.
        curr_pos, curr_quater = p.getBasePositionAndOrientation(item_id)
        curr_euler = np.array(p.getEulerFromQuaternion(curr_quater))
        #-- Stability (boolean) is determined by checking if the item has settled within a small tolerance of its target position and orientation. 
        stability = np.linalg.norm(new_pos-curr_pos)<0.02 and curr_euler.dot(target_euler)/(np.linalg.norm(curr_euler)*np.linalg.norm(target_euler)) > np.pi*2/3
        self.packed.append(item_id)
        self.unpacked.remove(item_id)
        
        return stability
            
    def drawAABB(self, aabb, width=1):
        #-- Function to draw a BBox in simulation
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
        
    def draw_box(self, width=5):
        #-- Function to draw the Box of the package in simulation
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
        #-- Function that creates the CSV file with the object number, name and path
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
            print('Total number of objects: ', tot_num_objects )

        return tot_num_objects
    
    def Compactness(self, item_in_box, item_volumes, box_hm):
        #-- Function that computes compactness 
        total_volume = 0
        for i,_ in enumerate(item_in_box):
            total_volume += item_volumes[i]
        box_volume = np.max(box_hm)*self.box_size[0]*self.box_size[1]
        return total_volume/box_volume

    def Pyramidality(self, item_in_box, item_volumes, box_hm):
        #-- Function that computes piramidality 
        total_volume = 0
        for i,item in enumerate(item_in_box):
            total_volume += item_volumes[i]
        used_volume = np.sum(box_hm)
        return total_volume/used_volume

if __name__ == '__main__':

    #-- Path with the URDF files
    obj_folder_path = '/Project/Irregular-Object-Packing/objects/'
    
    #-- PyBullet Environment setup 
    env = Env(obj_dir = obj_folder_path, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 40)

    #-- Generate csv file with objects 
    tot_num_objects = env.generate_urdf_csv()

    #-- Draw Box
    env.draw_box( width=5)

    #-- Load items 
    item_numbers = np.arange(55,57)
    item_ids = env.load_items(item_numbers)

    for i in range(500):
        p.stepSimulation()


    #-- Compute bounding boxes Items Volumes and orders them  
    volume_bbox, bbox_order = env.order_by_bbox_volume(env.loaded_ids)
    print(' The bbox_order by volume is: ', bbox_order)
        
    #-- Compute Item Volume
    item_volume = env.item_volume(item_ids[0])
    print(' The bbox_order by volume is: ', item_volume)
        
    #-- Compute Item HeightMap and visualize it
    item_id = item_ids[0]
    orient = [0, 90, 0]
    
    Ht,Hb = env.item_hm(item_id, orient )
    print(' The item heightmap shape are Ht: ', Ht.shape, ' Hb: ', Hb.shape )
    
    env.visualize_object_heightmaps(item_id, orient)
    env.visualize_object_heightmaps_3d(item_id, orient)

    for i in range(500):
        p.stepSimulation()

    #-- Pack items with random transformation
    for i,item_ in enumerate(item_ids):
        target_euler = [0,0,0]
        target_pos = [20-i*5,20-i*5,0] # cm
        transform = np.empty(6,)
        transform[0:3] = target_euler
        transform[3:6] = target_pos
        stability = env.pack_item(item_ , transform)
        print(' Is the placement stable?', stability)
    
    #-- Compute Box HeightMap
    BoxHeightMap = env.box_heightmap()
    print(' The box heightmap shape is: ', BoxHeightMap.shape)

    for i in range(500):
        p.stepSimulation()

    #-- Compute Piramidality and Compactness
    item_in_box = env.pack_item

    #-- Compute  Items Volumes and orders them  
    volume_items, volume_order = env.order_by_item_volume(env.packed)
    print(' The bbox_order by volume is: ', volume_order)

    C = env.Compactness(env.packed, volume_items, BoxHeightMap)
    print('Compactness is: ', C)
    P = env.Pyramidality(env.packed, volume_items, BoxHeightMap)   
    print('Piramidality is: ', P)

    env.visualize_box_heightmap()
    env.visualize_box_heightmap_3d()
    
    for i in range(500):
        p.stepSimulation()

    #-- Draw item BBOX in simulation 
    env.drawAABB(p.getAABB(item_id), width=1)


    #-- Remove all items 
    item_ids = env.remove_all_items()

    for i in range(500):
        p.stepSimulation()


    



