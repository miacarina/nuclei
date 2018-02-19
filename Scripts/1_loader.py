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
import sys

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import skimage.io

# Import custom functions
sys.path.append('/Users/Kaggle/nuclei/Functions')
from Image_Pre_Processing.Transform import rgb2gray

# Import parallelisation modules
import multiprocessing
from joblib import Parallel, delayed


import random


def image_id(full_image_folder_path, idx):
    image_id = os.path.basename(full_image_folder_path)
    return idx, image_id


def create_mask():
    
    return



train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

full_image_folder_paths = glob.glob(train_x_dir+'/*')

"""
random.seed(42)
all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
test_image_path = random.sample(all_image_paths,1)[0]
test_image_mask_paths = glob.glob(os.path.join('/'.join(test_image_path.split('/')[:-2]), 'masks', '*'))
"""


#bin boolean numpy array
#
#numpy array
#nuclei labels
#
#turn boolean numpy rray 
#rle - run length encoding 
#





test_image_path = '/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png'
test_image_mask_paths = glob.glob(os.path.join('/'.join(test_image_path.split('/')[:-2]), 'masks', '*'))


mask_path = test_image_mask_paths[0]
mask_image = skimage.io.imread(mask_path)


# Plot
plt.imshow(mask_image, 'gray')
plt.colorbar()
plt.show()

#mask_image = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,1,1]], dtype = 'uint8')
#
mask_image = mask_image.T

output = []
for i in range(0,len(mask_image)):
    if mask_image[i].sum() != 0:
        #print(i, mask_image[i])
        for k in range(len(mask_image)):
            if mask_image[i][k] !=0:
                print((i*len(mask_image)+k)+1, mask_image[i][k])
                output.append((i*len(mask_image)+k)+1)
                
        
from itertools import groupby, count

groupby(output, lambda n, c=count(): n-next(c))


def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0} {1}'.format(l[0], len(l))
    else:
        return '{0} {1}'.format(l[0], '1')

' '.join(as_range(g) for _, g in groupby(output, key=lambda n, c=count(): n-next(c)))

        