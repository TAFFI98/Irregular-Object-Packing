# Irregular-Object-Packing
Planning Irregular Object Packing via Hierarchical Reinforcement Learning.

This repo contains the basic code for Irregular Object Packing on PyBullet. 
This is an old version of our code and we are keeping updating.

## Quick Start

The code is based on PyTorch. Please prepare the object files from YCB and OCRTOC and put them into “objects” folder. Run env.generate() to generate index of object files. And then run main.py to train the model. 

 
## Repo organization 

The repo is organized as follows:
-	main.py: The main 
-	env.py: Functions to interact with the PyBullet simulation environment
-   models.py: Models for sequence planning and placement planning
-   trainer.py: Trainer for models
-   heuristics_HM.py: HM heuristic for placement
-   evaluate.py: Functions to evaluate performance of different methods on C, P and S.

- Link for downloading objects:
  - YCB:
	- https://github.com/eleramp/pybullet-object-models
  - OCRTOC:
    - http://www.ocrtoc.org
    
## Useful conda commands
- Create conda env from yml file
```
conda env create -f environment.yml
```
- See available environments in conda
```
conda env list
```
- export and environment in yml file
```
conda env export > environment.yml
```
          
## Install torch

- Install torch in the RL_exam environment according to your device setup: https://pytorch.org/get-started/locally/

For CPU
```
conda activate RL_exam
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
For CUDA
```
conda activate RL_exam
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
