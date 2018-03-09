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
from Image_Pre_Processing.Transform import rgb2grey
from General.Utility import flatten


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
plt.imshow(mask_image)
plt.colorbar()
plt.show()

#mask_image = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,1,1]], dtype = 'uint8')
#mask_image = final_array
mask_image = mask_image.T

print('Check for image that is more long than wide and see what code does... probably needs another exception, ha!')

output = []
if mask_image.shape[0] != mask_image.shape[1]:
    print('Not square.')
    mask_image = np.asmatrix(mask_image)
    
    for i in range(0, mask_image.shape[1]*mask_image.shape[0]):
#        print(i,mask_image.item(i))
        if mask_image.item(i) !=0:
            output.append(i+1)
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



"""
# Simple testing
re = np.array([[0, 0, 1, 1, 0], [0, 1, 1, 1, 0],[0, 0, 1, 0, 0]])

reverse = '5 1 7 5'

reverse_img = re

"""



def pairwise_grouping(ls):
    grouped_ls = []
    for n in range(0,int(len(ls)/2)):
        grouped_ls.append([int(ls[2*n]), int(ls[2*n+1])])
    return grouped_ls


def expand_ls(ls_expand):
    new_ls = []
    first = ls_expand[0]
    second = ls_expand[1]
    
    while second > 0:
        new_ls.append([first, 1])
        first = first + 1
        second = second - 1
    return new_ls



# 
reverse = '101 33 461 32 822 31 1182 31 1543 29 1903 29 2264 27 2624 26 2985 24 3346 23 3706 23 4066 22 4427 20 4788 18 5150 13 5511 11 5873 5'


img_path = '/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png'


# Reverse image plot for control
reverse_img = skimage.io.imread(img_path)
plt.imshow(reverse_img)
plt.colorbar()
plt.show()


ls = reverse.split(' ')


grouped_ls = pairwise_grouping(ls)


extended_ls = []
for item in grouped_ls:
    extended_ls.append(expand_ls(item))


final_ls = pairwise_grouping(flatten(extended_ls))



reshaped_matrix= np.zeros(shape = (reverse_img.shape[0], reverse_img.shape[1]), dtype=int)

for l in range(len(final_ls)):
    print(final_ls[l])
    print((final_ls[l][0]//reshaped_matrix.shape[0]),  (final_ls[l][0]%reshaped_matrix.shape[0]))
    reshaped_matrix[    (final_ls[l][0]//reshaped_matrix.shape[0]),  (final_ls[l][0]%reshaped_matrix.shape[0])   ] = 1



final_array = np.asarray(reshaped_matrix.T, dtype=np.uint8)



plt.imshow(final_array)
plt.show()





#if mask_image.shape[0] != mask_image.shape[1]:
#    mask_image = np.asmatrix(mask_image)
#    
#    for i in range(0, mask_image.shape[1]*mask_image.shape[0]):
#        print(i,mask_image.item(i))
#        if mask_image.item(i) !=0:
#            print(i)
#            output.append(i)
#    for i in range(0, mask_image.shape[0]):
#        if mask_image[i].sum() != 0:
#            print(i, mask_image[i])
#            for k in range(mask_image.shape[1]):
#                if mask_image[i][k] !=0 :
#                    print((2*(1+i), mask_image[i][k]))
#                    output.append((i*len(mask_image)+k))



