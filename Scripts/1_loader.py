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


def image_id(full_image_folder_paths):
    
    return idx, image_id


def create_mask():
    
    return



train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

full_image_folder_paths = glob.glob(train_x_dir+'/*')



output



