#!/bin/bash

conda create -n colmapenv python=3.7 -y
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
conda install pytorch==1.12.1 -c pytorch -c conda-forge -y
conda config --set channel_priority false
conda install colmap -c conda-forge -y