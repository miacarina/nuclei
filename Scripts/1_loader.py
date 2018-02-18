#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 18:42:36 2018

@author: Young
"""

# =============================================================================
# Loader
# =============================================================================

"""
Description:
Loads all training dataset into memory
"""


# Import modules
import os
import glob

import pandas as pd

import matplotlib.pyplot as plt

# Import parallelisation modules
import multiprocessing
from joblib import Parallel, delayed


import cv2



def image_id(full_image_folder_path, idx):
    image_id = os.path.basename(full_image_folder_path)
    return idx, image_id


def create_mask():
    
    return



train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

full_image_folder_paths = glob.glob(train_x_dir+'/*')

#bin boolean numpy array
#
#numpy array
#nuclei labels
#
#turn boolean numpy rray 
#rle - run length encoding 
#


os.path.splitext(
        
        
        os.path.basename(full_image_folder_paths[0])
        
        
        )



from skimage import data, io, filters

mask_path = '/Users/Kaggle/nuclei/Data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks/0adbf56cd182f784ca681396edc8b847b888b34762d48168c7812c79d145aa07.png'


j = skimage.io.imread(mask_path)

for i in range(0,len(u)):
    print(u[i].sum())
    
    
