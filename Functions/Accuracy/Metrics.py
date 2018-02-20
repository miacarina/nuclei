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
def plot_cm(y_test, preds, cmap=plt.cm.Blues):
    
    # Setup    
    classes     = np.unique(y_test)
    tick_marks  = np.arange(len(classes))
    cmap        = plt.cm.Blues
    
    # Confusion matrix
    cm          = confusion_matrix(y_test, preds)
    
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

