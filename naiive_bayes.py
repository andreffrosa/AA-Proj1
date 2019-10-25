#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:14:55 2019

@author: af.rosa
"""

import numpy as np
import sklearn.neighbors as sk
from sklearn.utils import check_X_y, check_array
import math

class naiive_bayes_classifier:
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.class_labels = None
        self.Priors = None
        self.kdes = None
        
    def fit(self, X, Y):
        """
        Fits the classifier given a matrix X (features) and a vector Y (Classes)
        """
        # Check X and Y
        X, Y = check_X_y(X, Y)
        
        # Obtain the labels of each class
        self.class_labels = np.array(list(set(Y)))
        
        n_features = X.shape[1]
        n_classes = len(self.class_labels)
        
        # Separate the training set into disjoint sets of each class and for each estimate the prior probability
        self.Priors = np.zeros(n_classes, dtype=np.float64)
        Classes = []
        for c in range(n_classes):
            Classes.append(X[Y == self.class_labels[c]])
            self.Priors[c] = math.log(float(Classes[c].shape[0]) / X.shape[0]) # Compute the log of the prior probability of belonging to (proportion of) class

        # For each feature of each class, compute the kernel density estimation to be used as the probability distribution of each feature (of each class)
        self.kdes = []
        for c in range(n_classes):
            class_kdes = []
            for feature in range(n_features):
                kde = sk.KernelDensity(bandwidth=self.bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
                kde.fit(Classes[c][:,feature].reshape(-1, 1))
                class_kdes.append(kde)
            self.kdes.append(class_kdes)
    
    def predict(self, X):
        """
        Predict the class of each example of X
        """
        # Check if X is valid and the state of the classifier
        X = check_array(X)
        self._check_fit(X)
       
        # Start with the prior probabilities (for each class)
        joint_log_probability = np.ones((X.shape[0], len(self.class_labels)), dtype=np.float64)
        joint_log_probability[:]*=self.Priors
        
        n_features = X.shape[1]
        n_classes = len(self.class_labels)
        
        # For each feature, add the log of the conditional probability (likelihood) of the feature given the class
        for c in range(n_classes):
            for feature in range(n_features):
                joint_log_probability[:,c] += self.kdes[c][feature].score_samples(X[:,feature].reshape((-1,1)))
        
        # Choose the class with highest probability as the class of each example
        predictions = np.zeros(X.shape[0])
        for ex in range(X.shape[0]):
            index = np.argmax(joint_log_probability[ex,:])
            if type(index) is not np.int64:
                index = index[0] # In case of a tie, choose the first
                
            predictions[ex] = self.class_labels[index]
            
        return predictions
    
    def _check_fit(self, X):
        """
        Verifies the state of the classifier
        """
        if self.kdes is None:
            raise ValueError('Classifiers is not Fit!')
        
        prev_features = len(self.kdes[0])
        if X.shape[1] != prev_features:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (X.shape[1], prev_features))
        
        if self.Priors is None:
            raise ValueError('Priors not assigned.')
        else:
            # Check that the sum is 1
            if not np.isclose(np.array(list(map(lambda x: math.exp(x), self.Priors))).sum(), 1.0):
                raise ValueError('The sum of the priors should be 1.')

            
        
