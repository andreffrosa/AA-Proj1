#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:10:28 2019

@author: af.rosa
"""

import numpy as np
import naiive_bayes as NB
import utils
import plots
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the train_data matrix and standardize it
train_data = np.loadtxt("TP1_train.tsv", delimiter='\t')
train_data,means,stdevs = utils.standardize(train_data);

# Load the test_data matrix and standardize it
test_data = np.loadtxt("TP1_test.tsv", delimiter='\t')
test_data,_,_ = utils.standardize(test_data, means=means, stdevs=stdevs)

#Shuffle train_data
np.random.shuffle(train_data)

classifiers = []

##############################################################################
# Naiive Bayes Classifier
 
def calc_bandwidth(index):
    return (index+1)*0.02

def calc_fold_nb(bandwidth, X, Y, train_ix, test_ix):
    nb = NB.naiive_bayes_classifier(bandwidth=bandwidth);
    nb.fit(X[train_ix,:], Y[train_ix])
    return nb.predict(X)

# Perform K-fold Cross Validation to find the best bandwidth value
best_index, errors = utils.cross_validate(10, train_data[:,:-1], train_data[:,-1], 30, calc_fold_nb, param_fun=calc_bandwidth, stratified=True, log=True)
best_bandwidth = errors[best_index,0]     

plots.plot_train_and_test_errors(errors[:,0], errors[:,1], errors[:,2], best_index, 'NB.png', 'Train', 'Validation', 'Trainning and Validation Errors','bandwidth', 'misclassifications (%)')

#Create and train a naiive bayes classifier
nb_clf = NB.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb_clf.fit(train_data[:,:-1], train_data[:,-1])

utils.add_classifier(classifiers, 'Naiive Bayes', nb_clf, train_data, test_data)

###########################################################################
# Gaussian Naiive Bayes Classifier

gnb_clf = GaussianNB()

utils.add_classifier(classifiers, 'Gaussian NB', gnb_clf, train_data, test_data)

############################################################################
# SVM (gamma) Classifier

def calc_gamma(index):
    return (index+1)*0.2

def calc_fold_svm(gamma, X, Y, train_ix, test_ix):
    clf = SVC(gamma=gamma, C=1.0)
    clf.fit(X[train_ix,:], Y[train_ix]) 
    return clf.predict(X)

best_index, errors = utils.cross_validate(10, train_data[:,:-1], train_data[:,-1], 30, calc_fold_svm, param_fun=calc_gamma, stratified=True, log=True)
best_gamma = errors[best_index,0]     

plots.plot_train_and_test_errors(errors[:,0], errors[:,1], errors[:,2], best_index, 'SVM.png', 'Train', 'Validation', 'Trainning and Validation Errors','gamma', 'misclassifications (%)')

#Create and train a SVM classifier
svm_clf = SVC(gamma=best_gamma, C=1.0)
utils.add_classifier(classifiers, 'SVM (gamma)', svm_clf, train_data, test_data)

############################################################################

# Compare Classifiers

print('approximate normal test')
print('name', '\t', 'train_err', '\t', 'test_err', '\t', 'confidence_interval')
for (name, clf, train_classifications, test_classifications) in classifiers:
    train_err = utils.compute_error(train_classifications, train_data[:,-1])
    test_err = utils.compute_error(test_classifications, test_data[:,-1])
    misclassified = sum(test_classifications != test_data[:,-1])
    
    ci = utils.confidence_interval(misclassified,test_data.shape[0])
    print(name, '\t', train_err, '\t', test_err, '\t', ci)

print('McNemar test')
print('clf1', 'vs', 'clf2', '=', 'value', ' (', 'perform_identically', ')')
for i in range(len(classifiers)):
    for j in range(len(classifiers)):
        if(i < j):
            print(i, j)
            (name1, clf1, train_classifications1, test_classifications1) = classifiers[i]
            (name2, clf2, train_classifications2, test_classifications2) = classifiers[j]
            
            perform_identically, value = utils.mc_nemar(test_classifications1, test_classifications2, test_data[:,-1])
            print(name1, 'vs', name2, '=', value, ' (', perform_identically, ')')

############################################################################


