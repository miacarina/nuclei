#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:28:12 2018

@author: Young
"""


# Plot confusion matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_cm(actuals, predictions, cmap=plt.cm.Blues):
    '''
    This function computers the confusion matrix and displays a pretty confusion matrix.

    Args:
        actuals (list of str/float/int): actual target values
        predictions (list of str/float/int): predictions of target values
        cmap (matplotlib colormap): colormap

    Returns:
        numpy array. confusion matrix
    '''

    # Setup    
    classes     = np.unique(actuals)
    tick_marks  = np.arange(len(classes))
    cmap        = plt.cm.Blues
    
    # Confusion matrix
    cm          = confusion_matrix(actuals, predictions)
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh      = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return cm

