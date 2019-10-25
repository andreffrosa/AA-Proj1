#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:04:04 2019

@author: andreffrosa
"""

import numpy as np
import utils
import naiive_bayes as NB
import naiive_bayes2 as NB2

# Load the train_data matrix and standardize it
train_data = np.loadtxt("TP1_train.tsv", delimiter='\t')
train_data,means,stdevs = utils.standardize(train_data);

# Load the test_data matrix and standardize it
test_data = np.loadtxt("TP1_test.tsv", delimiter='\t')
test_data,_,_ = utils.standardize(test_data, means=means, stdevs=stdevs)

#Shuffle train_data
np.random.shuffle(train_data)

best_bandwidth = 0.1

nb_clf = NB.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb_clf.fit(train_data[:,:-1], train_data[:,-1])
nb_clf_train_classifications = nb_clf.predict(train_data[:,:-1])
nb_clf_test_classifications =  nb_clf.predict(test_data[:,:-1])

nb_clf2 = NB2.naiive_bayes_classifier(bandwidth=best_bandwidth);
nb_clf2.fit(train_data[:,:-1], train_data[:,-1])
nb_clf2_train_classifications = nb_clf2.predict(train_data[:,:-1])
nb_clf2_test_classifications =  nb_clf2.predict(test_data[:,:-1])

print(sum(nb_clf_train_classifications == nb_clf2_train_classifications))
print(sum(nb_clf_test_classifications == nb_clf2_test_classifications))