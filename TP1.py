#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:10:28 2019

@author: af.rosa
"""

import numpy as np
import naiive_bayes

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

#Create and train a naiive bayes classifier
nb = naiive_bayes.naiive_bayes_classifier(bandwidth=1.0);
nb.fit(train_data)

# Load the test_data matrix and standardize it
test_data = np.loadtxt("TP1_test.tsv", delimiter='\t')
test_data,_,_ = standardize(test_data, means=means, stdevs=stdevs)

y_predicted = nb.predict(test_data)

misclassified = sum(test_data[:,-1] != y_predicted)

print(misclassified)


