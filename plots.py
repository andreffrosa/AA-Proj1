#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:07:21 2019

@author: af.rosa
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_train_and_test_errors(X, Y1, Y2, best, file, y1_key=None, y2_key=None, title=None, x_label=None, y_label=None):
    """
    Create image with plot for classifier
    """
    plt.figure()
    
    if(title is not None):
        plt.title(title)
    
    if(x_label is not None):
        plt.xlabel(x_label)
        
    if(y_label is not None):
        plt.ylabel(y_label)
        
    line, = plt.plot(X, Y1, 'b-')
    if(y1_key is not None):
        line.set_label(y1_key)

    line, = plt.plot(X, Y2, 'r-')
    if(y2_key is not None):
        line.set_label(y2_key)
    
    plt.axvline(x=X[best])
    
    plt.text(X[best]*1.05, min(Y1[best], Y2[best])*0.9, X[best])
    
    plt.legend()
    
    plt.savefig(file, dpi=400)
    plt.show()
    plt.close()
    
    
def plot_3d_test_errors(X, Y, Z, file):
    """
    Create image with plot for classifier
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    Y, X = np.meshgrid(Y, X)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(file, dpi=400)
    plt.show()
    plt.close()