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
# Utility Functions
# =============================================================================
def show_images(image_ids, train_x_dir):
  plt.close('all')  
  fig, ax = plt.subplots(nrows=len(image_ids),ncols=3, figsize=(10,10))

  for image_idx, image_id in enumerate(image_ids):
    image_path = os.path.join(train_x_dir, image_id, 'images', '{}.png'.format(image_id))
    mask_paths = os.path.join(train_x_dir, image_id, 'masks', '*.png')
  
    image = skimage.io.imread(image_path)
    masks = skimage.io.imread_collection(mask_paths).concatenate()
    mask = np.zeros(image.shape[:2], np.uint16)
    for mask_idx in range(masks.shape[0]):
      mask[masks[mask_idx] > 0] = mask_idx + 1
    other = mask == 0
    
    if len(image_ids) > 1:
      ax[image_idx, 0].imshow(image)
      ax[image_idx, 1].imshow(mask)
      ax[image_idx, 2].imshow(np.expand_dims(other, axis=2) * image)
    else:
      ax[0].imshow(image)
      ax[1].imshow(mask)
      ax[2].imshow(np.expand_dims(other, axis=2) * image)


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








