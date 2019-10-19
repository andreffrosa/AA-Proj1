#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:48:39 2019

@author: af.rosa
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold 

def identity(parameter):
    return parameter

def cross_validate(folds, X, Y, iterations, calc_fold, param_fun=identity, stratified=True, log=False):
    """
    """
    
    if(stratified):
        kf = StratifiedKFold(n_splits=folds)
    else:
        kf = KFold(n_splits=folds)

    errors = np.zeros((iterations, 3))
    best_index = -1

    if(log):
        print('parameter :\t', 'avg_train_error', '\t', 'avg_validation_error')

    for i in range(0, iterations):
        parameter = param_fun(i)
        avg_train_error = avg_validation_error = 0
        
        for train_indexes,validation_indexes in kf.split(X,Y):
            train_error,validation_error = calc_fold(parameter,X,Y,train_indexes,validation_indexes)
            avg_train_error += train_error 
            avg_validation_error += validation_error 
            
        avg_train_error = avg_train_error/folds
        avg_validation_error = avg_validation_error/folds
        
        if(log):
            print(parameter,':\t', avg_train_error, '\t', avg_validation_error)
         
        errors[i,:] = [parameter, avg_train_error, avg_validation_error]
        
        if(avg_validation_error < errors[best_index,2] or best_index==-1):
            best_index = i

    return best_index, errors
    #return best_parameter, best_train_error, best_validation_error

