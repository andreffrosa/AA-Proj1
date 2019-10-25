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

###############################################################################
# Naiive Bayes Classifier
 
def calc_bandwidth(index):
    return (index+1)*0.02

def calc_fold_nb(bandwidth):
    return NB.naiive_bayes_classifier(bandwidth=bandwidth);

# Perform K-fold Cross Validation to find the best bandwidth value
bandwidth_iteraions = 30
best_index, params, train_errors, validation_errors = utils.cross_validate(5, train_data[:,:-1], train_data[:,-1], bandwidth_iteraions, calc_fold_nb, param_fun=calc_bandwidth, stratified=True, log=True)
best_bandwidth = params[best_index]     
plots.plot_train_and_test_errors(np.array(params), train_errors, validation_errors, best_index, 'NB.png', 'Train', 'Validation', 'Trainning and Validation Errors','bandwidth', 'misclassifications (%)')

#Create and train a naiive bayes classifier
nb_clf = NB.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb_clf.fit(train_data[:,:-1], train_data[:,-1])

utils.add_classifier(classifiers, 'Naiive Bayes', nb_clf, train_data, test_data)

###############################################################################
## Gaussian Naiive Bayes Classifier

gnb_clf = GaussianNB()
utils.add_classifier(classifiers, 'Gaussian NB', gnb_clf, train_data, test_data)

###############################################################################
# SVM (gamma) Classifier

def calc_gamma(index):
    return (index+1)*0.2

def calc_fold_svm(gamma):
    return SVC(gamma=gamma, C=1.0)

gamma_iterations = 30
best_index, params, train_errors, validation_errors = utils.cross_validate(5, train_data[:,:-1], train_data[:,-1], gamma_iterations, calc_fold_svm, param_fun=calc_gamma, stratified=True, log=True)
best_gamma = params[best_index]     
plots.plot_train_and_test_errors(np.array(params), train_errors, validation_errors, best_index, 'SVM.png', 'Train', 'Validation', 'Trainning and Validation Errors','gamma', 'misclassifications (%)')

#Create and train a SVM classifier
svm_clf = SVC(gamma=best_gamma, C=1.0)
utils.add_classifier(classifiers, 'SVM (gamma)', svm_clf, train_data, test_data)

###############################################################################
# SVM (gamma & C) Classifier

def calc_fold_svm_c(C):
    def f(gamma):
        return SVC(gamma=gamma, C=C)
    return f

gamma_iterations = 30
gamma_values = np.linspace(0.01,10,gamma_iterations) 
def calc_gamma2(index):
    return gamma_values[index]

error_matrix = []
c_iterations = 30
#c_values = np.linspace(0.01,100,c_iterations) 
c_values = np.logspace(-2,2,c_iterations, base=10.0)
def get_params(index):
    C = c_values[index]
    # For each C, find the best gamma
    best_index, params, train_errors, validation_errors = utils.cross_validate(5, train_data[:,:-1], train_data[:,-1], gamma_iterations, calc_fold_svm_c(C), param_fun=calc_gamma2, stratified=True, log=False)
    best_gamma = params[best_index] 
    error_matrix.append(validation_errors)
    return (C, best_gamma);

def calc_fold_svm2(params):
    C = params[0]
    gamma = params[1]
    return SVC(gamma=gamma, C=C)

best_index, params, train_errors, validation_errors = utils.cross_validate(5, train_data[:,:-1], train_data[:,-1], c_iterations, calc_fold_svm2, param_fun=get_params, stratified=True, log=True)
best_C = params[best_index][0] 
best_gamma = params[best_index][1]

#error_matrix = np.array(error_matrix)
#gamma_values = np.linspace(0.02, 0.6, 30)
#plots.plot_3d_test_errors(c_values, gamma_values.transpose(), error_matrix, 'SVM2.png')

print('best_gamma', best_gamma, 'best_C', best_C)

##Create and train a SVM classifiere
enhanced_svm_clf = SVC(gamma=best_gamma, C=best_C)
utils.add_classifier(classifiers, 'SVM (gamma&C)', enhanced_svm_clf, train_data, test_data)

###############################################################################
# Compare Classifiers

print('\nApproximate Normal Test')
print('Classifier', '\t', 'train error %   ', '\t', 'test error %           ', '\t', 'confidence_interval')
for (name, clf, train_classifications, test_classifications) in classifiers:
    train_err = utils.compute_error(train_classifications, train_data[:,-1])
    test_err = utils.compute_error(test_classifications, test_data[:,-1])
    misclassified = sum(test_classifications != test_data[:,-1])
    
    ci = utils.confidence_interval(misclassified,test_data.shape[0])
    print(name, '\t', train_err, '\t', test_err, '\t', ci)

print('\nMcNemar Test')
print('classifier1', 'vs', 'classifier2', '=', 'value', '\t', '(', 'perform_identically', ')')
for i in range(len(classifiers)):
    for j in range(len(classifiers)):
        if(i < j):
            (name1, clf1, train_classifications1, test_classifications1) = classifiers[i]
            (name2, clf2, train_classifications2, test_classifications2) = classifiers[j]
            
            perform_identically, value = utils.mc_nemar(test_classifications1, test_classifications2, test_data[:,-1])
            print(name1, 'vs', name2, '=', value, ' (', perform_identically, ')')

###############################################################################
