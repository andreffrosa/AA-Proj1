#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:27:15 2019

@author: andreffrosa
"""

import numpy as np 
from sklearn.utils import shuffle
mat = np.loadtxt(’dataset_90.txt’,delimiter=’,’) 
data = shuffle(mat) 5 Ys = data[:,0] 
Xs = data[:,1:] 7 means = np.mean(Xs,axis=0) 
stdevs = np.std(Xs,axis=0) 
Xs = (Xs-means)/stdevs

def poly_16features(X): 
    """Expand data polynomially""" 
    X_exp = np.zeros((X.shape[0],X.shape[1]+14)) 
    X_exp[:,:2] = X 
    X_exp[:,2] = X[:,0]*X[:,1] 
    X_exp[:,3] = X[:,0]**2 
    X_exp[:,4] = X[:,1]**2 
    #... rest of the expansion here 
    return X_exp

from sklearn.linear_model import LogisticRegression 

def calc_fold(feats, X,Y, train_ix,test_ix,C=1e12): 
    """return classification error for train and test sets""" 
    reg = LogisticRegression(penalty=’l2’,C=C, tol=1e-10) 
    reg.fit(X[train_ix,:feats],Y[train_ix,0]) 
    prob = reg.predict_proba(X[:,:feats])[:,1] 
    squares = (prob-Y[:,0])**2 
    return (np.mean(squares[train_ix]), 
            p.mean(squares[test_ix]))

from sklearn.model_selection import KFold 
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
kf = KFold(n_splits=4) 
for train, valid in kf.split(x): 
    print (train, valid)

from sklearn.model_selection import train_test_split, StratifiedKFold 

Xs=poly_16features(Xs) 
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys) 
folds = 10 
kf = StratifiedKFold(n_splits=folds) 

for feats in range(2,16):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold(feats,X_r,Y_r,tr_ix,va_ix)
        tr_err += r 
        va_err += v 
    print(feats,’:’, tr_err/folds,va_err/folds)