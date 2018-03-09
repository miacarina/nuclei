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


from itertools import groupby, count





# Extract image ID
def image_id(full_image_folder_path):
    image_id = os.path.basename(full_image_folder_path).split('.')
    return image_id[0]


# Iterate through list to create range
def as_range(iterable):
    l = list(iterable)
    if len(l) > 1:
        return '{0} {1}'.format(l[0], len(l))
    else:
        return '{0} {1}'.format(l[0], '1')


# Create RLE from mask image path
def create_rle(mask_img_path):
    
    # Read in image from file path to mask
    mask_image = skimage.io.imread(mask_img_path)
    
    # Transpose array
    mask_image = mask_image.T
    
    # Create empty list for pixel positions
    output = []
    
    # Check if image is square
    if mask_image.shape[0] != mask_image.shape[1]:
        
        # Reshape array into matrix to use matrix item function
        mask_image = np.asmatrix(mask_image)
        
        # Loop through every item in matrix and append if value != 0
        for i in range(0, mask_image.shape[1]*mask_image.shape[0]):
            if mask_image.item(i) !=0:
                output.append(i+1)
                
    # Condition for non-square images
    else:
        for i in range(0, mask_image.shape[1]):
            
            # Check if row sum is != 0 and append value
            if mask_image[i].sum() != 0:
                for k in range(mask_image.shape[1]):
                    if mask_image[i][k] !=0:
                        output.append((i*len(mask_image)+k)+1)
    
    # Create rfe by splitting out start value to individual nunmbers
    rle = ' '.join(as_range(g) for _, g in groupby(output, key = lambda n, c = count(): n-next(c)))

    return rle, image_id(mask_img_path)



def flatten_masks(all_image_paths):
    mask_paths = []
    for i in range(0,len(all_image_paths)):
        mask_paths.append(glob.glob(os.path.join('/'.join(all_image_paths[i].split('/')[:-2]), 'masks', '*')))
    return flatten(mask_paths)


def img_mask_id(all_image_paths):
    
    img_mask_ids = []
    for i in range(0,len(all_image_paths)):
        img_mask_ids.append(glob.glob(os.path.join('/'.join(all_image_paths[i].split('/')[:-2]), 'masks', '*')))
    return img_mask_ids


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


def reverse_rle(rle_string, image_path):
    print(image_path)
    print('\n')
    # Split string on spaces
    ls = rle_string.split(' ')
    
    # Grouping pixel position and count
    grouped_ls = pairwise_grouping(ls)

    # Expand list by pixel counts
    expanded_ls = []
    for item in grouped_ls:
        expanded_ls.append(expand_ls(item))

    # Flatten list of lists and group pairwise
    final_ls = pairwise_grouping(flatten(expanded_ls))

    # Read in image format
    reverse_img = rgb2grey(skimage.io.imread(image_path))

    try:
        # Set up empty matrix with shape of image
        reshaped_matrix = np.zeros(shape = (reverse_img.shape[0], reverse_img.shape[1]), dtype=int)
        
        for l in range(len(final_ls)):
    #        print((final_ls[l][0]//reshaped_matrix.shape[0]), (final_ls[l][0]%reshaped_matrix.shape[0]))
            reshaped_matrix[    (final_ls[l][0]//reshaped_matrix.shape[0]), (final_ls[l][0]%reshaped_matrix.shape[0])   ] = 1
        
        # Create final image array
        final_array = np.asarray(reshaped_matrix.T, dtype=np.uint8)
    except(IndexError):
        final_array = image_path

    return final_array



if __name__ == '__main__':
    
    # Folder path to training data
    train_x_dir = '/Users/Kaggle/nuclei/Data/stage1_train'
    train_y_path = '/Users/Kaggle/nuclei/Data/stage1_train_labels.csv'
    
    
    full_image_folder_paths = glob.glob(train_x_dir+'/*')
    
    
    # Extract image paths
    all_image_paths = glob.glob(os.path.join(train_x_dir, '*', 'images', '*'))
    
    all_mask_paths = flatten_masks(all_image_paths)
    
    all_mask_df = pd.DataFrame(all_mask_paths, columns = {'paths'})
    
    all_mask_df['idx'] = all_mask_df.index
    
    
    # Define # of cores
    ncores         = multiprocessing.cpu_count() - 1
    
    
    #==============================================================================
    # Mask extraction
    #==============================================================================
    
#    # Job review extraction
#    output_masks = Parallel(n_jobs = ncores)(delayed(create_rle)(path) for path in all_mask_df['paths'])
#    
#    print('Check for image that is more long than wide and see what code does... probably needs another exception, ha!')
#    
#    mask_df = pd.DataFrame(output_masks)
#    
#    mask_df.columns = ['rle', 'mask_id']
#    
#    mask_df.to_csv('/Users/mia/Desktop/mask_df.csv', index=False)
#    
    
    #==============================================================================
    # Mask extraction
    #==============================================================================
    
    labels_df = pd.read_csv(train_y_path)
        
    




    output_image_arrays = Parallel(n_jobs = ncores)(delayed(reverse_rle)(rle_string, '/Users/Kaggle/nuclei/Data/stage1_train/'+image_id+'/images/'+image_id+'.png') for rle_string, image_id in zip(labels_df['EncodedPixels'], labels_df['ImageId']))



plt.imshow(final_array)
plt.show()






