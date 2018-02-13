#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:37:13 2018

@author: Young
"""

# Import modules
import os
import sys
import glob
import random
import skimage.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

# Import custom functions
sys.path.append('/Users/Kaggle/nuclei/Functions')
from Image_Pre_Processing.Transform import rgb2gray

# Import image specific modules
from skimage.filters import try_all_threshold


# =============================================================================
# Discovery
# =============================================================================
# Define paths
data_dir = '/Users/Kaggle/nuclei/Data'
train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

# Pick random test image
random.seed(42)
all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
test_image_path = random.sample(all_image_paths,1)[0]
test_image_mask_paths = glob.glob(os.path.join('/'.join(test_image_path.split('/')[:-2]), 'masks', '*'))

# Load image
test_image = skimage.io.imread(test_image_path)
test_masks = skimage.io.imread_collection(test_image_mask_paths).concatenate()

# Convert image to grayscale array
gray_image      = rgb2gray(test_image)

# Combine masks
mask = np.zeros(gray_image.shape[:2], np.uint16)
for mask_idx in range(test_masks.shape[0]):
    mask[test_masks[mask_idx] > 0] = mask_idx + 1

# Plot
plt.imshow(gray_image, 'gray')
plt.show()
plt.imshow(mask)
plt.colorbar()
plt.show()

# Try thresholding methods for getting mask
fig, ax = try_all_threshold(gray_image, figsize=(12, 10), verbose=False)
plt.show()








