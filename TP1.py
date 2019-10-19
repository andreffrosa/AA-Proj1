#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:10:28 2019

@author: af.rosa
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold 
import naiive_bayes as NB

#Standardize the data
def standardize(data, means=None, stdevs=None): # ignores the last column (classification)
    if means is None:
        means = np.mean(data[:,:-1], axis=0)
    
    if stdevs is None:
        stdevs = np.std(data[:,:-1], axis=0)
        
    data[:,:-1] = (data[:,:-1] - means)/stdevs
    return data,means,stdevs

# Load the train_data matrix and standardize it
train_data = np.loadtxt("TP1_train.tsv", delimiter='\t')
train_data,means,stdevs = standardize(train_data);

#Shuffle train_data
np.random.shuffle(train_data)

#
    
def calc_fold(bandwidth, X, Y, train_ix, test_ix):
    """return classification error for train and test sets""" 
    nb = NB.naiive_bayes_classifier(bandwidth=bandwidth);
    nb.fit(X[train_ix,:], Y[train_ix])
    classifications = nb.classify(X)#[:,1] 
    
    misclassified_train = sum(classifications[train_ix] != Y[train_ix])
    misclassified_test = sum(classifications[test_ix] != Y[test_ix])
    
    error_perc_train = (float(misclassified_train)/(len(train_ix)))*100
    error_perc_test = (float(misclassified_test)/(len(test_ix)))*100
    
    return (error_perc_train, error_perc_test)

folds = 10 
kf = StratifiedKFold(n_splits=folds)


best_va_err = 101.0
best_tr_err = 0
best_bandwidth = 0

for i in range(0,30):
    bandwidth = (i+1)*0.02
    tr_err = va_err = 0
    
    for tr_ix,va_ix in kf.split(train_data[:,:-1],train_data[:,-1]):
        r,v = calc_fold(bandwidth,train_data[:,:-1],train_data[:,-1],tr_ix,va_ix)
        tr_err += r 
        va_err += v 
        
    tr_err = tr_err/folds
    va_err = va_err/folds
    print(bandwidth,':', tr_err, '\t', va_err)
    
    if(va_err < best_va_err):
        best_va_err = va_err
        best_tr_err = tr_err
        best_bandwidth = bandwidth

print('\n')
print(best_bandwidth,':', best_tr_err, '\t', best_va_err)      

#Create and train a naiive bayes classifier
nb = NB.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb.fit(train_data[:,:-1], train_data[:,-1])

clasification = nb.classify(train_data[:,:-1])

misclassified = sum(train_data[:,-1] != clasification)
train_err = (float(misclassified)/(train_data.shape[0]))*100
print(train_err)

# Load the test_data matrix and standardize it
test_data = np.loadtxt("TP1_test.tsv", delimiter='\t')
test_data,_,_ = standardize(test_data, means=means, stdevs=stdevs)

clasification = nb.classify(test_data[:,:-1])

misclassified = sum(test_data[:,-1] != clasification)
test_err = (float(misclassified)/(test_data.shape[0]))*100
print(test_err)


