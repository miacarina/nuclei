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
from itertools import groupby, count


def image_id(full_image_folder_path, idx):
    image_id = os.path.basename(full_image_folder_path)
    return idx, image_id


def create_mask():
    
    return

def as_range(iterable): # not sure how to do this part elegantly
    l = list(iterable)
    if len(l) > 1:
        return '{0} {1}'.format(l[0], len(l))
    else:
        return '{0} {1}'.format(l[0], '1')
    


train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'

full_image_folder_paths = glob.glob(train_x_dir+'/*')

"""
random.seed(42)
all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
test_image_path = random.sample(all_image_paths,1)[0]
test_image_mask_paths = glob.glob(os.path.join('/'.join(test_image_path.split('/')[:-2]), 'masks', '*'))
"""



# Not square image
test_image_path = '/Users/Kaggle/nuclei/Data/stage1_train/5ef4442e5b8b0b4cf824b61be4050dfd793d846e0a6800afa4425a2f66e91456/images/5ef4442e5b8b0b4cf824b61be4050dfd793d846e0a6800afa4425a2f66e91456.png'

# Square image
test_image_path = '/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png'

test_image_mask_paths = glob.glob(os.path.join('/'.join(test_image_path.split('/')[:-2]), 'masks', '*'))


mask_path = test_image_mask_paths[0]
mask_image = skimage.io.imread(mask_path)


# Plot
plt.imshow(mask_image, 'gray')
plt.colorbar()
plt.show()

#mask_image = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,1,1]], dtype = 'uint8')


mask_image = mask_image.T

print('Check for image that is more long than wide and see what code does... probably needs another exception, ha!')

output = []
if mask_image.shape[0] != mask_image.shape[1]:
    print('Not square.')
#    mask_image = np.asmatrix(mask_image)
#    
#    for i in range(0, mask_image.shape[1]*mask_image.shape[0]):
##        print(i,mask_image.item(i))
#        if mask_image.item(i) !=0:
#            output.append(i)
    
    for i in range(0, mask_image.shape[0]):
        if mask_image[i].sum() != 0:
            print(i, mask_image[i])
            for k in range(mask_image.shape[0]):
                if mask_image[i][k] !=0 :
                    print((2*(1+i), mask_image[i][k]))
                    output.append((i*len(mask_image)+k))
else:
    print('Square.')
    for i in range(0, mask_image.shape[1]):
        if mask_image[i].sum() != 0:
#            print(i, mask_image[i])
            for k in range(mask_image.shape[1]):
                if mask_image[i][k] !=0:
#                    print((i*len(mask_image)+k)+1, mask_image[i][k])
                    output.append((i*len(mask_image)+k)+1)



' '.join(as_range(g) for _, g in groupby(output, key=lambda n, c=count(): n-next(c)))





reverse_img = '/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png'

reverse_img = skimage.io.imread(reverse_img)


reverse = '214863 14 215378 23 215896 27 216413 31 216932 35 217451 38 217970 42 218489 45 219008 49 219528 53 220047 63 220566 67 221086 70 221605 73 222125 74 222645 75 223165 76 223685 76 224205 77 224725 77 225245 78 225765 78 226285 78 226805 78 227326 77 227846 76 228367 74 228887 73 229408 72 229929 70 230450 68 230971 66 231492 64 232013 61 232534 58 233055 54 233577 49 234104 40 234631 32 235154 24'



reverse_img.shape




reverse_array = np.array([[0,0,1,1,0],[0,1,1,1,0],[0,0,1,0,0]])

if mask_image.shape[0] != mask_image.shape[1]:
    mask_image = np.asmatrix(mask_image)
    
    for i in range(0, mask_image.shape[1]*mask_image.shape[0]):
        print(i,mask_image.item(i))
        if mask_image.item(i) !=0:
            print(i)
            output.append(i)






