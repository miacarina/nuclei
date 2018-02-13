#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:38:00 2018

@author: Young
"""

import numpy as np
from skimage import img_as_ubyte

def rgb2gray(rgb_image):
    return np.dot(img_as_ubyte(rgb_image)[...,:3], [0.299, 0.587, 0.114])