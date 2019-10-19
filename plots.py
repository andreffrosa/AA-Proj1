#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:07:21 2019

@author: andreffrosa
"""

import matplotlib.pyplot as plt

def plot_train_and_test_errors(X, Y1, Y2, best, file, y1_key=None, y2_key=None, title=None, x_label=None, y_label=None):
    """create imege with plot for best classifier"""
    #ax_lims=(0,0.7,0,100)
    #plt.figure(figsize=(8,8), frameon=False)
    plt.figure()
    
    if(title is not None):
        plt.title(title)
    
    if(x_label is not None):
        plt.xlabel(x_label)
        
    if(y_label is not None):
        plt.ylabel(y_label)
        
    #plt.axis(ax_lims)
    #reg = LogisticRegression(C=best_c, tol=1e-10)
    #reg.fit(X_r,Y_r)
    #plotX,plotY,Z = poly_mat(reg,X_r,16,ax_lims)
    #plt.plot(X,Y1, colors = ('b', 'r'),alpha=0.5)
    #plt.plot(X,Y2, colors = ('g', 'r'),alpha=0.5)
    line, = plt.plot(X, Y1, 'b-')
    if(y1_key is not None):
        line.set_label(y1_key)

    line, = plt.plot(X, Y2, 'r-')
    if(y2_key is not None):
        line.set_label(y2_key)
    
    plt.axvline(x=X[best])
    
    plt.text(X[best]*1.05, min(Y1[best], Y2[best])*0.9, X[best])
    
    plt.legend()
    
    #plt.contour(plotX,plotY,Z,[0], colors = ('k'))
   # plt.plot(X_r[Y_r>0,0],X_r[Y_r>0,1],'or')
    #plt.plot(X_r[Y_r<=0,0],X_r[Y_r<=0,1],'ob')
    #plt.plot(X_t[Y_t>0,0],X_t[Y_t>0,1],'xr',mew=2)
    #plt.plot(X_t[Y_t<=0,0],X_t[Y_t<=0,1],'xb',mew=2)
    plt.savefig(file, dpi=400)
    plt.show()
    plt.close()