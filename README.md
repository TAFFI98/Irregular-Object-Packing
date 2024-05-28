# Irregular-Object-Packing
Planning Irregular Object Packing via Hierarchical Reinforcement Learning.

This repo contains the basic code for Irregular Object Packing on PyBullet. 

The code is based on PyTorch. Please prepare the object files from YCB and OCRTOC and put them into `objects` folder. In this repo there are still some objects in the folder. Use them as guideline for format requirements.

Link for downloading additional objects:
  - YCB:
	- https://github.com/eleramp/pybullet-object-models
  - OCRTOC:
    - http://www.ocrtoc.org
    
 
## Repo organization 

The repo is organized as follows:

-	`main_module.py`: The main file script.
-	`env.py`: Functions to interact with the PyBullet simulation environment.
- `models.py`: Models for sequence planning and placement planning.
- `trainer.py`: Trainer class to train models.
- `tester.py`: Tester class to test models.
- `heuristics_HM.py`: HM heuristic class for placement.
- `environment_GPU.yml`: yml file for conda environment for GPU support (NVIDIA-SMI 535.161.08 - Driver Version: 535.161.08 - CUDA Version: 12.2).
- `environment_noGPU.yml`: yml file for conda environment for CPU support.
- `fonts_terminal.py`: Colors definition for terminal prints.
- `pybullet_test.py`: File to test pybullet installation.
- `visualize_models.py`: File to visualize the models structure using graphviz.


## Useful conda commands
- Create conda env from yml file
```
conda env create -f environment_GPU.yml
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

Guidelines to install torch in the conda environment according to your device setup: https://pytorch.org/get-started/locally/

# Quick Start

1. The first thing to do is to create a conda environment containig the packages listed in `environment_GPU.yml` and `environment_noGPU.yml` respectively if you want to use your GPU to train/test the models or not. You can use the **Useful conda commands** to do that.

2. You can use the file `main_module.py` to train or test the models.

**Train options**: 

- `--obj_folder_path`: path to the folder containing the objects .csv file
- `--gui`: GUI for PyBullet
- `--force_cpu`: Use CPU instead of GPU
- `--stage`: stage 1 or 2 for training
- `--k_max`: Max number of objects to load
- `--k_min`: Min number of objects to load
- `--k_sort`: Number of objects to consider for sorting
- `--manager_snapshot`: path to the manager network snapshot
- `--worker_snapshot`:  path to the worker network snapshot
- `--new_episodes`: number of episodes
- `--load_snapshot_manager`: Load snapshot of the manager network
- `--load_snapshot_worker`: Load snapshot of the worker network