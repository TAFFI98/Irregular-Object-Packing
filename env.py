
from fonts_terminal import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pybullet as p
import time
import pybullet_data
import os, csv
import random
import gc
import math
import trimesh

'''
This class defines the environment for the packing problem. It contains the following methods:
- __init__: initializes the environment
- create_box: creates the walls of the packing box in simulation
- load_items: loads the objects in simulation
- remove_all_items: removes all the objects from the simulation
- box_heightmap: computes the box heightmap
- visualize_object_heightmaps: visualizes the object heightmaps in 2D
- visualize_object_heightmaps_3d: visualizes the object heightmaps in 3D
- visualize_box_heightmap: visualizes the box heightmap in 2D
- visualize_box_heightmap_3d: visualizes the box heightmap in 3D
- compute_object_limits: computes the object limits
- stop_physics: stops the physics of the simulation
- item_hm: computes the item heightmap
- order_by_bbox_volume: orders the items by bounding box volume
- compute_object_bbox: computes the object bounding box
- order_by_item_volume: orders the items by volume
- grid_scan: scans the grid
- pack_item_check_collision: checks the collision of the packed item
- check_height_exceeded: checks if the height is exceeded
- bounding_box_collision: checks the bounding box collision between box and objects
- unpack_item: unpacks the item
- drawAABB: draws the bounding box
- ray_cast_in_batches: casts the ray in batches
- item_volume: computes the item volume
- compute_object_volume: computes the object volume
- draw_predicted_pose_volume: draws the predicted pose volume
- removeAABB: removes the AABB
- draw_box: draws the box
- Compactness: computes the compactness
- Piramidality: computes the piramidality
- Objective_function: computes the objective function
- generate_urdf_csv: generates the urdf csv
- Reward_function: computes the reward function
- get_z: gets the z
- euler_angle_distance: computes the euler angle distance

'''
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
        wall_1_index = self.create_box([boxL,wall_width,boxH],[boxL/2,-(wall_width+0.002)/2,boxH/2])
        wall_2_index = self.create_box([boxL,wall_width,boxH],[boxL/2,boxW+(wall_width+0.002)/2,boxH/2])
        wall_3_index = self.create_box([wall_width,boxW,boxH],[-(wall_width+0.002)/2,boxW/2,boxH/2])
        wall_4_index = self.create_box([wall_width,boxW,boxH],[boxL+(wall_width+0.002)/2,boxW/2,boxH/2])
        self.walls_indices = [wall_1_index, wall_2_index, wall_3_index, wall_4_index]
        
        #-- Set mass to 0 to avoid box movements
        for i in range(4):
            p.changeDynamics(self.walls_indices[i], -1, mass=0) # Set mass to 0 to avoid box movements
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=-180, cameraPitch=-89, cameraTargetPosition=[0.15,0.07,-0.04])
        p.stepSimulation()
        #-- Initialize ids of the objects loaded in simulation
        self.loaded_ids = []
        #-- Initialize ids of the objects already packed
        self.packed = []
        #-- Initialize ids of the objects not packed
        self.unpacked = []  
        # -- Initialize the stability tresholds
        self.stability_treshold_position = 0.02
        self.stability_treshold_orientation = np.pi/6
        
    def create_box(self, size, pos):
        '''
        size: list of 3 elements [length, width, height]
        pos: list of 3 elements [x, y, z]
        output: integer representing the index of the box in the simulation
        
        Function to create the walls of the packing box in simulation
        '''
        size = np.array(size)
        shift = [0, 0, 0]
        beige = [144/255, 87/255, 35/255, 1]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        rgbaColor=beige,
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
        print('Loading', len(item_ids) ,' objects in simluation:')
        # Read URDF information from CSV file
        with open(self.csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            urdf_info = list(reader)

        for count,id in enumerate(item_ids):
            position = [i for i, d in enumerate(urdf_info) if int(d['Index']) == id]
            urdf_path = urdf_info[position[0]]['URDF Path']
            scale = random.uniform(0.8, 1.2)  # Generate a random scale factor
            loaded_id = p.loadURDF(urdf_path, [(count//5)/4+2.2, (count%5)/4+0.2, 0.1], flags=flags, globalScaling=scale)
            print('- ', count+1 ,'.',urdf_path.split('/')[-2] , '(', urdf_path.split('/')[-3], ', scale:', np.round(scale,3), ', id:',loaded_id,')')
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
        
    def box_heightmap(self, batch_size=10000):
        '''
        output: numpy array of shape (resolution, resolution)

        Function to compute the box heigthmap
        '''
        sep = self.box_size[0]/self.resolution
        xpos = np.arange(sep/2,self.box_size[0]+sep/2,sep)
        ypos = np.arange(sep/2,self.box_size[1]+sep/2,sep)
        xscan, yscan = np.meshgrid(xpos,ypos)
        ScanArray = np.array([yscan.reshape(-1),xscan.reshape(-1)])
        Start = np.insert(ScanArray,2,self.box_size[2],0).T
        End = np.insert(ScanArray,2,0,0).T

        # Split rays into batches
        num_batches = int(np.ceil(len(Start) / batch_size))
        RayScan = []
        for i in range(num_batches):
            start_batch = Start[i*batch_size:(i+1)*batch_size]
            end_batch = End[i*batch_size:(i+1)*batch_size]
            RayScan.extend(p.rayTestBatch(start_batch, end_batch))

        RayScan = np.array(RayScan, dtype=object)
        Height = (1-RayScan[:,2].astype('float64'))*self.box_size[2]
        HeightMap = Height.reshape(self.resolution,self.resolution).T
        return HeightMap
    
    def visualize_object_heightmaps(self, Ht, Hb, orient, only_top = False):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        only_top: boolean
        
        Function to visualize the object heightmaps (bottom and top) in 2D
        '''
        
        # Display the top-view and bottom-view heightmaps
        if only_top == False:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(Ht, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].set_title(f"Rotated Top Heightmap (orient={orient}°)")
            axs[1].imshow(Hb, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            axs[1].set_title(f"Rotated Bottom Heightmap (orient={orient}°)")
            plt.tight_layout()
            plt.show()
        if only_top == True:
            # Display the top-view  heightmap
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
            axs.imshow(Ht, cmap='viridis', origin='lower', extent=[0, self.box_size[0], 0, self.box_size[1]])
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_title(f"Heightmap (orient={orient}°)")
            plt.tight_layout()
            plt.show()

    def visualize_object_heightmaps_3d(self, Ht, Hb, orient, only_top = False ):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        only_top: boolean
        
        Function to visualize the object heightmaps (bottom and top) in 3D
        '''

        # Create 3D grid
        X, Y = np.meshgrid(np.linspace(0, self.box_size[0], Ht.shape[1]),
                        np.linspace(0, self.box_size[1], Ht.shape[0]))
        if only_top == False:
            # Plot the top-view heightmap
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_surface(X, Y, Ht, cmap='viridis')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Height')
            ax1.set_title(f"Rotated Top Heightmap (orient={orient}°)")

            # Plot the bottom-view heightmap
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(X, Y, Hb, cmap='viridis')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Height')
            ax2.set_title(f"Rotated Bottom Heightmap (orient={orient}°)")
            plt.tight_layout()
            plt.show()

        if only_top == True:
            # Plot the box heightmap
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Ht, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Height')
            ax.set_title(f"Heightmap (orient={orient}°)")
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

    def compute_object_limits(self, pos, offsets):
        '''
        pos: list of 3 elements [x, y, z]
        offsets: list of 6 elements [offset_pointminx_COM, offset_pointmaxx_COM, offset_pointminy_COM, offset_pointmaxy_COM, offset_pointminz_COM, offset_pointmaxz_COM]
        output: list of 6 elements [min_x_global, max_x_global, min_y_global, max_y_global, min_z_global, max_z_global]
        This function computes the limits of the object in the global reference frame
        '''

        offset_pointminx_COM, offset_pointmaxx_COM, offset_pointminy_COM, offset_pointmaxy_COM, offset_pointminz_COM, offset_pointmaxz_COM  =   offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5]

        min_x_global = pos[0] - offset_pointminx_COM
        max_x_global = pos[0] + offset_pointmaxx_COM
        
        min_y_global = pos[1] - offset_pointminy_COM
        max_y_global = pos[1] + offset_pointmaxy_COM

        min_z_global = pos[2] - offset_pointminz_COM
        max_z_global = pos[2] + offset_pointmaxz_COM

        object_limits = [min_x_global, max_x_global, min_y_global, max_y_global, min_z_global, max_z_global]

        return object_limits
    
    def stop_physics(self):
        '''
        Function to stop the physics of the simulation
        '''
        for body_id in range(p.getNumBodies()):
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
            for joint_id in range(p.getNumJoints(body_id)):
                p.setJointMotorControl2(body_id, joint_id, p.VELOCITY_CONTROL, force=0)

    def item_hm(self, item_id, orient):
        '''
        item_id: integer
        orient: list of 3 elements [roll, pitch, yaw]
        output: numpy array of shape (resolution, resolution), numpy array of shape (resolution, resolution), float, float, float
        Ht: numpy array of shape (resolution, resolution) representing the top heightmap
        Hb: numpy array of shape (resolution, resolution) representing the bottom heightmap
        obj_length: float representing the length of the object
        obj_width: float representing the width of the object
        offsets: list of 6 elements [offset_pointminx_COM, offset_pointmaxx_COM, offset_pointminy_COM, offset_pointmaxy_COM, offset_pointminz_COM, offset_pointmaxz_COM]
        '''
        
        # Stores the item original position and orientation
        old_pos, old_quater = p.getBasePositionAndOrientation(int(item_id))  
        quater = p.getQuaternionFromEuler(orient)
        old_vel = p.getBaseVelocity(int(item_id))
        # Set new orientation for the item
        p.resetBasePositionAndOrientation(int(item_id),[1,1,1],quater)       
        self.stop_physics()
        _, vertices = p.getMeshData(bodyUniqueId=int(item_id), linkIndex=-1)

        # vertices is a list of tuples, each representing a vertex of the mesh.
        # We can use a list comprehension to get all the z values and then find the maximum.

        # Get the rotation matrix from the quaternion orientation
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quater)).reshape(3, 3)

        # Transform the vertices to the global reference frame
        global_vertices = [np.dot(rotation_matrix, vertex) + np.array([1,1,1]) for vertex in vertices]

        # Extract the min and max values of the global vertices
        min_z_global = min(vertex[2] for vertex in global_vertices)        
        min_x_global = min(vertex[0] for vertex in global_vertices)
        min_y_global = min(vertex[1] for vertex in global_vertices)
        max_z_global = max(vertex[2] for vertex in global_vertices)        
        max_x_global = max(vertex[0] for vertex in global_vertices)
        max_y_global = max(vertex[1] for vertex in global_vertices)
        del(global_vertices)
        # Extracts the boundingbox of the item
        offset_pointminz_COM = 1 - min_z_global
        offset_pointmaxz_COM = max_z_global -  1

        offset_pointminx_COM = 1 - min_x_global
        offset_pointmaxx_COM = max_x_global -1

        offset_pointminy_COM = 1 - min_y_global
        offset_pointmaxy_COM = max_y_global -1

        obj_length = offset_pointmaxx_COM + offset_pointminx_COM
        obj_width = offset_pointmaxy_COM + offset_pointminy_COM

        # Computation of a grid of points within the bounding box of the item. 
        # Computation of a grid of points within the specified box size
        sep = self.box_size[0] / self.resolution
        xpos = np.arange(sep / 2, self.box_size[0] + sep / 2, sep)
        ypos = np.arange(sep / 2, self.box_size[1] + sep / 2, sep)
        xscan, yscan = np.meshgrid(xpos, ypos)
        
        # Adjust the position of the grid to center the object in the 2D image
        x_offset = (self.box_size[0] - obj_length) / 2
        y_offset = (self.box_size[1] - obj_width) / 2
        xscan += min_x_global - x_offset
        yscan += min_y_global - y_offset

        # Ray casting from the top to the bottom and from the bottom to the top of the item's bounding box 
        # to find the height of points on the surface of the bounding box.
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Top = np.insert(ScanArray,2,max_z_global,axis=0).T
        Down = np.insert(ScanArray,2,min_z_global,axis=0).T
        RayScanTD = self.ray_cast_in_batches(Top, Down)
        RayScanDT = self.ray_cast_in_batches(Down, Top)
        
        # Computes the heights of points on the top (Ht) and bottom (Hb) surfaces of the bounding box based on the results of the ray casting.
        Ht = (1-RayScanTD[:,2])*(max_z_global - min_z_global)
        Hb = (1-RayScanDT[:,2])*(max_z_global - min_z_global)

        Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
        Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
        

        offsets = [offset_pointminx_COM, offset_pointmaxx_COM, offset_pointminy_COM, offset_pointmaxy_COM, offset_pointminz_COM, offset_pointmaxz_COM]

        # Resets the initial orientation for the item
        p.resetBasePositionAndOrientation(int(item_id),old_pos,old_quater)
        p.resetBaseVelocity(int(item_id), old_vel[0], old_vel[1])
        return Ht,Hb, obj_length, obj_width, offsets 
        
    def ray_cast_in_batches(self, starts, ends, batch_size=10000):
        '''
        starts: numpy array of shape (n,3)
        ends: numpy array of shape (n,3)
        batch_size: integer
        output: numpy array of shape (n,3)
        This function casts rays in batches to avoid memory issues.
        '''
        results = []
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i+batch_size]
            batch_ends = ends[i:i+batch_size]
            results.extend(p.rayTestBatch(batch_starts, batch_ends))
        return np.array(results, dtype=object)

    def order_by_bbox_volume(self,items_ids):
        '''
        items_ids: list of integers
        output: list of floats (bounding boxes volumes), list of integers (ordered ids)
        
        Function to determine the order of objects based on their bounding box volumes in descending order.
        '''
        volume = []
        for item in items_ids:
            AABB = np.array(p.getAABB(int(item) ))
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
        
        Function to compute the volume of an item.
        ''' 
        old_pos, old_quater = p.getBasePositionAndOrientation(item)
        
        # Retrieve visual shape data
        visual_shape_data = p.getVisualShapeData(item)

        if len(visual_shape_data) == 0:
            print("No visual shape data found for the item.")
            return 0
        
        volume = 0
        
        for shape_data in visual_shape_data:
            # Check if the shape type is mesh
            if shape_data[2] == p.GEOM_MESH:
                file_path = shape_data[4].decode('utf-8')
                if not os.path.isabs(file_path):
                    # If the path is not absolute, we need to construct the full path
                    file_path = os.path.join(os.path.dirname(__file__), file_path)
                
                try:
                    # Load the mesh using trimesh from the file path
                    mesh = trimesh.load(file_path)
                    volume += mesh.volume
                except Exception as e:
                    continue
        
        # Reset the position and orientation of the item
        p.resetBasePositionAndOrientation(item, old_pos, old_quater)
        
        return volume
    
    def pack_item_check_collision(self, item_id, transform, offsets):
        '''
        item_id: integer representing the id of the object
        transform: list of 6 elements [target_euler, target_pos]
        offsets: list of 6 elements [offset_pointminx_COM, offset_pointmaxx_COM, offset_pointminy_COM, offset_pointmaxy_COM, offset_pointminz_COM, offset_pointmaxz_COM]
        outputs:
        NewBoxHeightMap: numpy array of shape (resolution, resolution) representing the new box heightmap
        stability: integer representing the stability of the item
        old_pos: list of 3 elements representing the old position of the item
        old_quater: list of 4 elements representing the old orientation of the item
        collision: boolean indicating if the item collides with the box margins
        limits_obj_line_ids: list of integers representing the line ids
        height_exceeded_before_pack: boolean indicating if the height of the box is exceeded before packing the item


        Function to reset the position and orientation of the item based on the provided transform argument[target_euler,target_pos]. It updates the environment attributes (env.unpacked and env.packed) accordingly.
        '''
        # Store the original position and orientation of the item
        old_pos, old_quater = p.getBasePositionAndOrientation(int(item_id) )
        
        # Check if the item is already packed
        if item_id in self.packed:
            print(f" {red}Item", int(item_id) , f"already packed!{exit}")
            return False
        # Extract the target position and orientation from the transform argument
        target_euler = transform[0:3]
        target_pos = transform[3:6]
        new_quater = p.getQuaternionFromEuler(target_euler)
 
        # Check collision with box margins 
        limits_object = self.compute_object_limits(target_pos, offsets)
            
        # Show the bounding box of the object to check collisions visullay
        limits_obj_line_ids = self.draw_predicted_pose_volume(limits_object)
        collision, height_exceeded_before_pack = self.bounding_box_collision(self.bbox_box, limits_object)

        # Set the new position and orientation for the item
        p.resetBasePositionAndOrientation(int(item_id) , target_pos, new_quater)

        # Checks the stability of the item based on its final position and orientation
        for i in range(500):
            p.stepSimulation()
            
        
        AABB_after_pack = self.drawAABB( self.compute_object_bbox(int(item_id) ))
        curr_pos, curr_quater = p.getBasePositionAndOrientation(int(item_id) )
        curr_euler = np.array(p.getEulerFromQuaternion(curr_quater))
        
        #-- Stability (boolean) is determined by checking if the item has settled within a small tolerance of its target position and orientation. 

        stability_bool = np.linalg.norm(target_pos-curr_pos) < self.stability_treshold_position and self.euler_angle_distance(curr_euler,target_euler) > self.stability_treshold_orientation
        print('Orientation error:', self.euler_angle_distance(curr_euler,target_euler))
        print('Position error:', np.linalg.norm(target_pos-curr_pos))

        stability = 1 if stability_bool else 0        
        
        # Update the packed and unpacked lists
        self.packed.append(int(item_id) )
        self.unpacked.remove(int(item_id) )
        self.removeAABB(AABB_after_pack)
        # Compute the new box heightmap
        NewBoxHeightMap = self.box_heightmap()
        
        return NewBoxHeightMap, stability, old_pos, old_quater,  collision, limits_obj_line_ids, height_exceeded_before_pack
    
    def check_height_exceeded(self,box_heightmap, box_height):
        '''
        box_heightmap: heightmap of the box
        box_height: height of the box
        output: boolean indicating if the height of the box is exceeded
        '''
        # Compute the new heightmap after placing the object
        max_z = box_heightmap.max()

        # Check if any height in the new heightmap exceeds the box height
        return max_z > box_height
    
    def bounding_box_collision(self, box_box, box_obj):
        '''
        box1: bounding box of the box
        box2: bounding box of the object
        
        output: boolean indicating if the bounding box of the object is entirely inside the bounding box of the box (sensibility: mm)
        '''
        # Each box is a tuple of (min_x, min_y, min_z, max_x, max_y, max_z)
        min_x_box, min_y_box, _, max_x_box, max_y_box, box_height = [round(x, 3) for x in box_box]
        min_x_obj, max_x_obj, min_y_obj, max_y_obj, max_z_obj = round(box_obj[0], 3), round(box_obj[1], 3), round(box_obj[2], 3), round(box_obj[3], 3), round(box_obj[5], 3)

        # Check if box2 is entirely inside box1
        collision = not (min_x_box <= min_x_obj and max_x_box >= max_x_obj and min_y_box <= min_y_obj and max_y_box >= max_y_obj )
        height_exceeded_before_pack = not (max_z_obj <= box_height)

        return collision, height_exceeded_before_pack
    
    def unpack_item(self, item_id, old_pos, old_quater):
        '''
        item_id: integer representing the id of the object
        old_pos: list of 3 elements representing the old position of the item
        old_quater: list of 4 elements representing the old orientation of the item

        Restores the position and orientation of the item to its original state and updates env (env.packed and env.packed) attributes accordingly.
        '''
        p.resetBasePositionAndOrientation(int(item_id) ,old_pos,old_quater)
        self.packed.remove(int(item_id))
        self.unpacked.append(int(item_id))
        NewBoxHeightmap = self.box_heightmap()
        return NewBoxHeightmap
         
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

    def draw_predicted_pose_volume(self, vertices, line_color=[1, 0, 0], line_width=2):
        '''
        vertices: list of 6 elements [X1, X2, Y1, Y2, Z1, Z2]
        line_color: RGB color of the line
        line_width: width of the line
        This function draws the predicted pose of the object in the simulation
        '''
        X1, X2, Y1, Y2, Z1, Z2 = vertices

        # Define the four corners of the rectangle
        point1 = [X1, Y1, Z1]
        point2 = [X2, Y1, Z1]
        point3 = [X2, Y2, Z1]
        point4 = [X1, Y2, Z1]

        # Draw the four sides of the rectangle
        line_ids = []
        line_id = p.addUserDebugLine(point1, point2, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point2, point3, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point3, point4, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point4, point1, line_color, line_width)
        line_ids.append(line_id)

        # Define the four corners of the rectangle
        point1_bis = [X1, Y1, Z2]
        point2_bis = [X2, Y1, Z2]
        point3_bis = [X2, Y2, Z2]
        point4_bis = [X1, Y2, Z2]

        # Draw the four sides of the rectangle
        line_id = p.addUserDebugLine(point1_bis, point2_bis, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point2_bis, point3_bis, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point3_bis, point4_bis, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point4_bis, point1_bis, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point1_bis, point1, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point2_bis, point2, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point3_bis, point3, line_color, line_width)
        line_ids.append(line_id)
        line_id = p.addUserDebugLine(point4_bis, point4, line_color, line_width)
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
            print('Total number of different available objects: ', tot_num_objects )

        return tot_num_objects
    
    def Compactness(self, item_in_box, item_volumes, box_hm):
        '''
        item_in_box: list of integers representing the ids of the items in the box
        item_volumes: list of floats representing the volumes of the items
        box_hm: numpy array of shape (resolution, resolution) representing the box heightmap
        output: float representing the compactness of the packing

        Function to compute the compactness of the packing
        '''
        if not item_in_box:
                return 0
        else:
            total_volume = 0
            for i,_ in enumerate(item_in_box):
                total_volume += item_volumes[i]
            box_volume = np.max(box_hm)*self.box_size[0]*self.box_size[1]
            print(f'{purple_light}>Compactness is: ', total_volume/box_volume, f'{reset}')
            return total_volume/box_volume

    def Pyramidality(self, item_in_box, item_volumes, box_hm):
        '''
        item_in_box: list of integers representing the ids of the items in the box
        item_volumes: list of floats representing the volumes of the items
        box_hm: numpy array of shape (resolution, resolution) representing the box heightmap
        output: float representing the piramidality of the packing
        
        Function to compute the piramidality of the packing
        '''
        if not item_in_box:
                return 0
        else:
            total_volume = 0
            for i,item in enumerate(item_in_box):
                total_volume += item_volumes[i]
            used_volume = np.sum(box_hm)
            print(f'{purple_light}>Piramidality is: ', total_volume/used_volume, f'{reset}')
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
        print(f'---------------------------------------') 
        print(f"{blue_light}\n5. Computing the objective function {reset}")
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
        return reward
    

    def calculate_reward(is_packed, attempt, max_attempts):
        """
        Calcola la ricompensa per il tentativo di inserimento dell'oggetto.

        Parametri:
        - is_packed: boolean, True se l'oggetto è stato inserito correttamente, False altrimenti
        - attempt: int, numero del tentativo corrente
        - max_attempts: int, numero massimo di tentativi consentiti

        Output:
        - reward: float, il valore della ricompensa per questo tentativo
        """
        
        # Se l'oggetto non è stato inserito
        if not is_packed:
            return -1
        
        # Se l'oggetto è stato inserito
        if is_packed:
            # Se è stato inserito al primo tentativo, la ricompensa massima è 1
            if attempt == 1:
                return 1
            # Riduci la ricompensa in base al numero di tentativi
            else:
                reward = 1 - (attempt / max_attempts)
                return reward
    
    def get_z(self, Hc, Hb, pixel_x, pixel_y, obj_length, obj_width):
        '''
        Hc: numpy array of shape (resolution, resolution) representing the current box heightmap
        Hb: numpy array of shape (resolution, resolution) representing the heightmap of the item
        pixel_x: integer representing the x-coordinate of the pixel representing the item position in the box
        pixel_y: integer representing the y-coordinate of the pixel representing the item position in the box
        obj_length: float representing the length of the item
        obj_width: float representing the width of the item

        output: float representing the maximum z-value of the item in the box
        
        Function to compute the maximum z-value of the item in the box due to gravity
        '''
        # Select the region of Hc and Hb that corresponds to the item
        x_range = [pixel_x + i for i in range(math.ceil(-self.resolution*obj_width/self.box_size[1]/2), math.ceil(self.resolution*obj_width/self.box_size[1]/2) )]
        y_range = [pixel_y + i for i in range(math.ceil(-self.resolution*obj_length/self.box_size[1]/2), math.ceil(self.resolution*obj_length/self.box_size[1]/2) )]
        x_min, x_max, y_min, y_max  = max(0,x_range[0]), min(Hb.shape[0],x_range[-1]), max(0,y_range[0]), min(Hb.shape[1],y_range[-1])
        if y_max - y_min == 0 or x_max - x_min == 0:
            y_max = y_min + 1
            x_max = x_min + 1
        Hc = Hc[x_min:x_max, y_min:y_max]

        # Initialize an empty list to store the results
        diffs = []

        # Determine the chunk size
        chunk_size = 100

        # Calculate the number of chunks
        num_chunks = np.prod(Hb.shape) // chunk_size
        
        # Iterate over the chunks
        for chunk_num in range(num_chunks):
            # Calculate the start and end indices for this chunk
            start_index = chunk_num * chunk_size
            end_index = (chunk_num + 1) * chunk_size

            # Get the chunk of Hb
            Hb_chunk = Hb.reshape(-1)[start_index:end_index].reshape(-1, 1, 1)

            # Subtract the chunk from Hc and append the result to the list
            diffs.append(np.max(Hc.reshape(1, 1, *Hc.shape) - Hb_chunk))

        # Handle the last chunk separately, if there is one
        if np.prod(Hb.shape) % chunk_size != 0:
            start_index = num_chunks * chunk_size
            end_index = np.prod(Hb.shape)
            Hb_chunk = Hb.reshape(-1)[start_index:end_index].reshape(-1, 1, 1)
            
            diffs.append(np.max(Hc.reshape(1, 1, *Hc.shape) - Hb_chunk))

        del( Hc, Hb, Hb_chunk)
        gc.collect()

        # Find the maximum value among all subtractions
        max_z = max(diffs)

        ''' Uncomment to visualize the heightmaps and the pixel position'''
        # fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        # axs.imshow(Hc, cmap='viridis', origin='lower')
        # axs.plot(pixel_x, pixel_y, 'ro')
        # axs.plot([max(0,x_range[0]), min(Hb.shape[0],x_range[-1])], [pixel_y, pixel_y], 'r-')
        # axs.plot( [pixel_x, pixel_x],[max(0,y_range[0]), min(Hb.shape[1],y_range[-1])], 'r-')
        # axs.set_xlabel('X')
        # axs.set_ylabel('Y')
        # plt.tight_layout()
        # plt.show()


        ''' -------- Old way to compute max_z with for loops ---not efficient ''' 
        #s_range = range(0, Hb.shape[0])
        #t_range = range(0, Hb.shape[1])
        #max_z = 0
        #ss_range = range(-(int(self.resolution*obj_length//self.box_size[0]))//2, int((self.resolution*obj_length//self.box_size[0]))//2)
        #tt_range = range(-(int(self.resolution*obj_width//self.box_size[1]))//2, int((self.resolution*obj_width//self.box_size[1]))//2)
        #for s in s_range:
        #         for t in t_range:
        #                 for ss in ss_range:
        #                     for tt in tt_range:
        #                         if 0 <= pixel_x+ss < Hc.shape[0] and 0 <= pixel_y+tt < Hc.shape[1]:
        #                             z = Hc[pixel_x+ss, pixel_y+tt] - Hb[s, t]
        #                             max_z = max(max_z, z)
        #max_z = max((Hc[pixel_x+ss, pixel_y+tt] - Hb[s, t] for s in s_range for t in t_range for ss in ss_range for tt in tt_range if 0 <= pixel_x+ss < Hc.shape[0] and 0 <= pixel_y+tt < Hc.shape[1]), default=0)

        return max_z
    
    def euler_angle_distance(self, curr_euler, target_euler):
        """
        Calculate the metric expressing the distance between two Euler angles.
        
        Parameters:
        curr_euler (np.ndarray): Current Euler angles (roll, pitch, yaw).
        target_euler (np.ndarray): Target Euler angles (roll, pitch, yaw).
        
        Returns:
        float: A value representing the angle between the two Euler angle vectors in radians.
            If one of the vectors is zero, returns the norm of the non-zero vector.
            If both vectors are zero, returns 0.
        """
        # Ensure the inputs are numpy arrays
        curr_euler = np.array(curr_euler)
        target_euler = np.array(target_euler)
        
        # Calculate the magnitudes (norms) of the Euler angle vectors
        curr_norm = np.linalg.norm(curr_euler)
        target_norm = np.linalg.norm(target_euler)
        
        # Handle cases where one or both norms are zero
        if curr_norm == 0 and target_norm == 0:
            return 0.0  # Both vectors are zero, the angle distance is zero
        elif curr_norm == 0:
            return target_norm  # The distance is the magnitude of the target vector
        elif target_norm == 0:
            return curr_norm  # The distance is the magnitude of the current vector
        
        # Calculate the dot product
        dot_product = np.dot(curr_euler, target_euler)
        
        # Calculate the cosine of the angle between the two vectors
        cosine_similarity = dot_product / (curr_norm * target_norm)
        
        # Clamp the cosine similarity value to the range [-1, 1] to avoid numerical errors
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # Calculate the angle (in radians) between the two vectors
        angle = np.arccos(cosine_similarity)
        
        return angle