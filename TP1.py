#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:10:28 2019

@author: af.rosa
"""

import numpy as np
import naiive_bayes as NB
import k_fold
import plots
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

# Load the test_data matrix and standardize it
test_data = np.loadtxt("TP1_test.tsv", delimiter='\t')
test_data,_,_ = standardize(test_data, means=means, stdevs=stdevs)

#Shuffle train_data
np.random.shuffle(train_data)

# Perform K-fold Cross Validation to find the best bandwidth value
 
def calc_bandwidth(index):
    return (index+1)*0.02

def calc_fold_nb(bandwidth, X, Y, train_ix, test_ix):
    """return classification error for train and test sets""" 
    nb = NB.naiive_bayes_classifier(bandwidth=bandwidth);
    nb.fit(X[train_ix,:], Y[train_ix])
    classifications = nb.classify(X)
    return classifications

best_index, errors = k_fold.cross_validate(10, train_data[:,:-1], train_data[:,-1], 30, calc_fold_nb, param_fun=calc_bandwidth, stratified=True, log=True)

best_bandwidth = errors[best_index,0]
best_tr_err = errors[best_index,1]
best_va_err = errors[best_index,2]

print('\n')
print(best_bandwidth,':', best_tr_err, '\t', best_va_err)      

plots.plot_train_and_test_errors(errors[:,0], errors[:,1], errors[:,2], best_index, 'NB.png', 'Train', 'Validation', 'Trainning and Validation Errors','bandwidth', 'misclassifications (%)')

#Create and train a naiive bayes classifier
nb = NB.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb.fit(train_data[:,:-1], train_data[:,-1])

classifications = nb.classify(train_data[:,:-1])
train_err = k_fold.compute_error(classifications, train_data[:,-1])
print(train_err)

classifications = nb.classify(test_data[:,:-1])
test_err = k_fold.compute_error(classifications, test_data[:,-1])
print(test_err)

###########################################################################

clf = GaussianNB()
clf.fit(train_data[:,:-1], train_data[:,-1])

classifications = clf.predict(train_data[:,:-1])
train_err = k_fold.compute_error(classifications, train_data[:,-1])
print(train_err)

classification = clf.classify(test_data[:,:-1])
test_err = k_fold.compute_error(classifications, test_data[:,-1])
print(test_err)

############################################################################

def calc_gamma(index):
    return (index+1)*0.2

def calc_fold_svm(gamma, X, Y, train_ix, test_ix):
    """return classification error for train and test sets""" 
    clf = SVC(gamma=gamma, C=1.0)
    clf.fit(X[train_ix,:], Y[train_ix]) 
    classifications = clf.predict(X)
    return classifications

best_index, errors = k_fold.cross_validate(10, train_data[:,:-1], train_data[:,-1], 30, calc_fold_svm, param_fun=calc_gamma, stratified=True, log=True)

best_gamma = errors[best_index,0]
best_tr_err = errors[best_index,1]
best_va_err = errors[best_index,2]

print('\n')
print(best_gamma,':', best_tr_err, '\t', best_va_err)      

plots.plot_train_and_test_errors(errors[:,0], errors[:,1], errors[:,2], best_index, 'SVM.png', 'Train', 'Validation', 'Trainning and Validation Errors','bandwidth', 'misclassifications (%)')

#Create and train a SVM classifier
clf = SVC(gamma=best_gamma, C=1.0)
clf.fit(train_data[:,:-1], train_data[:,-1]) 

classifications = clf.predict(train_data[:,:-1])
train_err = k_fold.compute_error(classifications, train_data[:,-1])
print(train_err)

classification = clf.classify(test_data[:,:-1])
test_err = k_fold.compute_error(classifications, test_data[:,-1])
print(test_err)