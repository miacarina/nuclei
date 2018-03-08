#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:21:08 2018

@author: Young
"""


# =============================================================================
# Import Modules
# =============================================================================
# Import general modules
import os
import sys
import glob
import random
import skimage.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom functions
sys.path.append('/Users/Kaggle/nuclei/Functions')
from Image_Pre_Processing.Transform import rgb2gray

# Import image specific modules
from skimage.filters import try_all_threshold


# =============================================================================
# Utility Functions
# =============================================================================
def load_image(image_path):
    '''
    This function takes the full path of an image (.png) file and loads in the 
    grey image as well as the masks
    
    Args:
        image_path (str): full file path of image (.png) file

    Returns:
        2d numpy array (int): greyscale image
        2d numpy array (int): image of masks with unique label for individual masks
    '''
    # Get corresponding mask paths
    image_mask_paths = glob.glob(os.path.join('/'.join(image_path.split('/')[:-2]), 'masks', '*'))
    
    # Load image
    image = skimage.io.imread(image_path)
    masks = skimage.io.imread_collection(image_mask_paths).concatenate()
    
    # Convert image to grayscale array
    grey_image      = rgb2gray(image)
    
    # Combine masks
    mask = np.zeros(gray_image.shape[:2], np.uint16)
    for mask_idx in range(masks.shape[0]):
        mask[masks[mask_idx] > 0] = mask_idx + 1
    return grey_image, mask

# Define paths
data_dir = '/Users/Kaggle/nuclei/Data'
train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

# Pick random test image
random.seed(42)
all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
test_image_path = random.sample(all_image_paths,1)[0]

grey_image, mask = load_image(test_image_path)

plt.imshow(mask)

np.unique(grey_image)


