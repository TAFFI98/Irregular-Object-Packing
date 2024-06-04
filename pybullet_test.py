import pybullet as p
import pybullet_data
import time
import os
#-- PyBullet Environment setup
physicsClient = p.connect(p.GUI)
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

for i in range (10000):
   p.stepSimulation()
   time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)

p.disconnect()