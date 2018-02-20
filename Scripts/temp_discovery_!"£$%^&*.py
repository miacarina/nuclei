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
from General.Utility import flatten
from Accuracy.Metrics import plot_cm
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

from skimage import measure
from skimage.filters import median
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.morphology import disk

# Threshold
threshold_function = threshold_otsu
#threshold_function = threshold_mean
thresh = threshold_function(gray_image)
threshold_mask = gray_image > thresh

# Smoothing
smoothing_shape_size = 3
smoothed_threshold_mask = median(threshold_mask, selem = disk(smoothing_shape_size))

# Invert the image if object is on the darker end
if(np.sum(smoothed_threshold_mask==255)>np.sum(smoothed_threshold_mask==0)):
    smoothed_threshold_mask = 255 - smoothed_threshold_mask

# Segment to connected objects
auto_segmented_smoothed_threshold_mask = measure.label(smoothed_threshold_mask)

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max

distance = ndi.distance_transform_edt(smoothed_threshold_mask)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((4, 4)),
                            labels=smoothed_threshold_mask)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=smoothed_threshold_mask)
plt.imshow(smoothed_threshold_mask)
plt.show()
plt.imshow(labels)
plt.show()
len(np.unique(labels))



from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
img = img_as_float(test_image[:,:,:3])
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

plt.imshow(felzenszwalb(smoothed_threshold_mask, scale=20, sigma=1, min_size=20))
plt.colorbar()

watershed(gradient, markers=250, compactness=0.001)

# =============================================================================
# Plot
# =============================================================================
# Image with correct masks
figsize_shape = (4,4)

plt.figure(figsize = figsize_shape)
plt.imshow(gray_image, 'gray')
plt.title('Original Image')
plt.show()

# Image with correct masks
plt.figure(figsize = figsize_shape)
plt.imshow(gray_image * (mask>0), 'gray' )
plt.title('Image + correct masks')
plt.show()

# Threshold
plt.figure(figsize = figsize_shape)
plt.imshow(gray_image * (threshold_mask>0), 'gray')
plt.title('Image + threshold mask')
plt.show()

# Threshold smoothed
plt.figure(figsize = figsize_shape)
plt.imshow(gray_image * (smoothed_threshold_mask>0), 'gray')
plt.title('Image + smooth threshold mask')
plt.show()

# Segmented threshold smoothed
plt.figure(figsize = figsize_shape)
plt.imshow(auto_segmented_smoothed_threshold_mask)
plt.colorbar()
plt.title('Image + segmented smooth threshold mask')
plt.show()




# =============================================================================
# Mask accuracy evaluator
# =============================================================================


# =============================================================================
# # Define Utility Function
# =============================================================================
def mask_binarize_flatten(mask):
    mask_binarized = mask*1 > 0
    mask_binarized_flattened = flatten(mask_binarized)
    return mask_binarized_flattened


# =============================================================================
# Main
# =============================================================================
# Define paths


# Loop over each actual mask



true_mask = mask
pred_mask = smoothed_threshold_mask

flattened_true_mask = mask_binarize_flatten(true_mask)
flattened_pred_mask = mask_binarize_flatten(pred_mask)

# Confusion matrix
from sklearn.metrics import classification_report
conf_mat = plot_cm(flattened_true_mask, flattened_pred_mask)
plt.show()
print(classification_report(flattened_true_mask, flattened_pred_mask))







# =============================================================================
# Test
# =============================================================================
#green channel happends to produce slightly better results
#than the grayscale image and other channels
img_gray=test_image[:,:,1]#cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#morphological opening (size tuned on training data)
circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
img_open=cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
#Otsu thresholding
img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
#Invert the image in case the objects of interest are in the dark side
if(np.sum(img_th==255)>np.sum(img_th==0)):
    img_th=cv2.bitwise_not(img_th)
#second morphological opening (on binary image this time)
bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
#connected components
cc=cv2.connectedComponents(bin_open)[1]
#cc=segment_on_dt(bin_open,20)

plt.imshow(bin_open)

from skimage import measure
plt.imshow(measure.label(bin_open))


import os
import cv2

def process(img_rgb):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    img_gray=img_rgb[:,:,1]#cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc


test_connected_components=[process(test_image)]
plt.imshow(test_connected_components[0])
plt.colorbar()

def rle_encoding(cc):
    values=list(np.unique(cc))
    values.remove(0)
    RLEs=[]
    for v in values:
        dots = np.where(cc.T.flatten() == v)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        RLEs.append(run_lengths)
    return RLEs

test_RLEs=[rle_encoding(cc) for cc in test_connected_components]

test_dirs = os.listdir('/Users/Kaggle/nuclei/Data/stage1_test')
test_dirs = [a for a in test_dirs if a != '.DS_Store']

with open('/Users/Kaggle/nuclei/Submission/test.csv', "a") as myfile:
    myfile.write('ImageId,EncodedPixels\n')
    for i,RLEs in enumerate(test_RLEs):
        for RLE in RLEs:
            myfile.write(test_dirs[i]+","+" ".join([str(i) for i in RLE])+"\n")