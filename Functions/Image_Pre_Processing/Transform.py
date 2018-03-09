#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:38:00 2018

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

# Import image specific modules
from skimage import img_as_ubyte
from skimage.filters import try_all_threshold


# =============================================================================
# Utility Functions
# =============================================================================
def rgb2grey(rgb_image_array):
    '''
    This function converts an RGB image (nxm array of 4-tuples) into greyscale.
    
    Args:
        numpy array n x m x 4: Image
    
    Returns:
        numpy array n x m: Image
        
    Example:
        >>> import PIL.Image as im
        >>> image_path = "/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png"
        >>> image_array = np.asarray(im.open(image_path))
        >>> rgb2grey(image_array)
    '''
    return np.dot(img_as_ubyte(rgb_image_array)[...,:3], [0.299, 0.587, 0.114])


def load_image(image_path):
    '''
    This function takes the full path of an image (.png) file and loads in the 
    grey image as well as the masks
    
    Args:
        image_path (str): full file path of image (.png) file

    Returns:
        2d numpy array (int): greyscale image
        2d numpy array (int): image of masks with unique label for individual masks
        
    Example:
        >>> image_path = "/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png"
        >>> grey_image, mask = load_image(image_path)
    '''
    # Get corresponding mask paths
    image_mask_paths = glob.glob(os.path.join('/'.join(image_path.split('/')[:-2]), 'masks', '*'))
    
    # Load image
    image = skimage.io.imread(image_path)
    masks = skimage.io.imread_collection(image_mask_paths).concatenate()
    
    # Convert image to grayscale array
    grey_image      = rgb2grey(image)
    
    # Combine masks
    mask = np.zeros(grey_image.shape[:2], np.uint16)
    for mask_idx in range(masks.shape[0]):
        mask[masks[mask_idx] > 0] = mask_idx + 1
    return grey_image, mask


def plot_mask_comparison(image, true_mask, prediction_mask, figsize = (10,10), cmap = None):
    '''
    This function plots 4 figures:
        1. Image overlayed with original mask
        2. Image overlayed with prediction mask
        3. True mask
        4. Prediction mask
        
    Args:
        image (numpy array of int): image of nuclei
        true_mask (numpy array of int): true masks collapsed into one array. This is an output of Transform.load_image
        prediction_mask (numpy array of int): prediction masks collapsed into one array. This is an output of Transform.load_image
        
    Returns: 
        Plot of the following 4 figures:
            1. Image overlayed with original mask
            2. Image overlayed with prediction mask
            3. True mask
            4. Prediction mask
    '''
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
    ax1.imshow(image * (true_mask > 0), cmap = cmap)
    ax1.set_title('Image with true mask')
    ax2.imshow(image * (prediction_mask > 0), cmap = cmap)
    ax2.set_title('Image with prediction mask')
    ax3.imshow(true_mask, cmap = cmap)
    ax3.set_title('True mask')
    ax4.imshow(prediction_mask, cmap = cmap)
    ax4.set_title('Prediction mask')
    plt.show()
    

def plot(image_array, title=None, figsize=(5,5), cmap='gray'):
    '''
    A function for lazy people to plot will minimal effort and limited functionality.
    
    Args:
        image_array (numpy array/PIL.Image object): image to be plotted
        title (str): title of image
    
    Returns:
        Plot of image.
        
    Example:
        >>> image_path = "/Users/Kaggle/nuclei/Data/stage1_train/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d/images/89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png"
        >>> grey_image, _ = load_image(image_path)
        >>> plot(grey_image)
    '''
    plt.figure(figsize = figsize)
    plt.imshow(image_array, cmap=cmap)
    if title:
        plt.title(title)
    plt.show()
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    