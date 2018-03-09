#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:59:39 2018

@author: Young
"""


# =============================================================================
# Import modules
# =============================================================================
# Import general modules
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
from General.Utility import flatten
from Accuracy.Metrics import plot_cm
from Image_Pre_Processing.Transform import plot
from Image_Pre_Processing.Transform import rgb2grey
from Image_Pre_Processing.Transform import load_image
from Image_Pre_Processing.Transform import plot_mask_comparison

# Import image specific modules
from skimage import measure
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import try_all_threshold



# =============================================================================
# Main
# =============================================================================
# Define paths
data_dir = '/Users/Kaggle/nuclei/Data'
train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

# Pick random test image
random.seed(42)
all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
test_image_path = random.sample(all_image_paths,1)[0]

# Load image and masks
grey_image, mask = load_image(test_image_path)    

# Threshold
threshold_function = threshold_otsu
#threshold_function = threshold_mean
thresh = threshold_function(grey_image)
threshold_bool = grey_image > thresh

# Smoothing
smoothing_shape_size = 3
smoothed_threshold_bool = median(threshold_bool, selem = disk(smoothing_shape_size))

# Invert the image if object is on the darker end
if (np.sum(smoothed_threshold_bool==255)>np.sum(smoothed_threshold_bool==0)):
    smoothed_threshold_bool = 255 - smoothed_threshold_bool

# Segment to connected objects
smoothed_threshold_mask = measure.label(smoothed_threshold_bool)

# Plot comparison of mask
plot_mask_comparison(grey_image, mask, smoothed_threshold_mask, cmap = 'gray')

## Other plots
#figsize = (5,5)
#plot(grey_image, title='Original image', figsize=figsize)
#plot(grey_image * (mask>0), title='Image + correct masks', figsize=figsize)
#plot(grey_image * (threshold_bool>0), title='Image + threshold mask', figsize=figsize)
#plot(grey_image * (smoothed_threshold_bool>0), title='Image + smooth threshold mask', figsize=figsize)
#plot(smoothed_threshold_mask, title='Image + segmented smooth threshold mask', figsize=figsize)




