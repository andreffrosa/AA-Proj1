#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:48:39 2019

@author: af.rosa
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold 
import math

def normalize(X):
    """
    Normalizes each column of X
    """
    min_value = X.min(axis=0)
    max_value = X.max(axis=0)
    return (X - min_value)/(max_value - min_value)

def standardize(data, means=None, stdevs=None):
    """
    Standardizes the data. Outputs standardized data, means and std deviation computed.
    """
    if means is None:
        means = np.mean(data[:,:-1], axis=0)
    
    if stdevs is None:
        stdevs = np.std(data[:,:-1], axis=0)
        
    data[:,:-1] = (data[:,:-1] - means)/stdevs
    return data,means,stdevs

def add_classifier(classifier_list, name, clf, train_data, test_data):
    """
    Adds classifier to the provided classifier list after training it and prediciting the classifications of the training set and the test set
    """
    clf.fit(train_data[:,:-1], train_data[:,-1])
    train_classifications = clf.predict(train_data[:,:-1])
    test_classifications =  clf.predict(test_data[:,:-1])
    classifier_list.append((name, clf, train_classifications, test_classifications))

def identity(parameter):
    """
    Returns parameter
    """
    return parameter

def compute_error(classifications, Y):
    """
    Computes the percentage of misclassifications.
    """
    misclassified = sum(classifications != Y)
    error_perc = (float(misclassified)/(Y.shape[0]))
    return error_perc

def cross_validate(folds, X, Y, iterations, calc_fold, param_fun=identity, stratified=True, log=False):
    """
    Cross validates a classifier to estimate the best parameter
    """
    if(stratified):
        kf = StratifiedKFold(n_splits=folds)
    else:
        kf = KFold(n_splits=folds)

    train_errors = np.zeros(iterations)
    validation_errors = np.zeros(iterations)
    params = []
    best_index = -1

    if(log):
        print('parameter :\t', 'avg_train_error', '\t', 'avg_validation_error')

    for i in range(0, iterations):
        parameter = param_fun(i)
        avg_train_error = avg_validation_error = 0
        
        for train_indexes,validation_indexes in kf.split(X,Y):
            clf = calc_fold(parameter)
            clf.fit(X[train_indexes,:], Y[train_indexes]) 
            classifications = clf.predict(X)
            avg_train_error += compute_error(classifications[train_indexes], Y[train_indexes])
            avg_validation_error += compute_error(classifications[validation_indexes], Y[validation_indexes])
            
        avg_train_error = avg_train_error/folds
        avg_validation_error = avg_validation_error/folds
        
        if(log):
            print('   ', parameter,'\t', avg_train_error, '\t', avg_validation_error)
         
        train_errors[i] = avg_train_error
        validation_errors[i] = avg_validation_error
        params.append(parameter)
        
        if(avg_validation_error < validation_errors[best_index] or best_index==-1):
            best_index = i

    return best_index, params, train_errors, validation_errors


def confidence_interval(X, N):
    """
    Returns the confidence interval for the expected number of erros of the classifier
    X = number of misclassiﬁed examples
    N = total size of the test set
    """
    p0 = float(X)/N
    std_dev = math.sqrt(N*p0*(1.0-p0))
    return (X-1.96*std_dev,X+1.96*std_dev)

def mc_nemar(clf1_predictions, clf2_predictions, Y):
    """
    performs the mcNemar test, returning true if the classifiers MAY perform identically.
    """
    e_01 = sum(np.logical_and((clf1_predictions != Y), (clf2_predictions == Y)))
    e_10 = sum(np.logical_and((clf2_predictions != Y), (clf1_predictions == Y)))
    
    value = ((float(abs(e_01 - e_10)) - 1.0)**2) / float((e_01 + e_10))
    
    return value <= 3.84, value






