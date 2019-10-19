#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:14:55 2019

@author: af.rosa
"""

import sklearn.neighbors as sk
import math

class naiive_bayes_classifier:
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.P_c0 = 0
        self.P_c1 = 0
        self.kdes_c0 = []
        self.kdes_c1 = []
        self.n_features = 0
        
    def fit(self, X, Y):
        """
        """
        # Separate the training set into disjoint sets of each class
        C0 = X[Y == 0.0]
        C1 = X[Y == 1.0]

        # Compute the log of the prior probability of belonging to (proportion of) each class
        self.P_c0 = math.log(C0.shape[0] / X.shape[0])
        self.P_c1 = math.log(C1.shape[0] / X.shape[0]) # = 1 - P-c0
        
        # Compute the number of features
        self.n_features = X.shape[1]
        
        # For each feature of each class, compute the kernel density estimation to be used as the probability distribution of each feature (of each class)
        for feature in range(self.n_features):
            kde = sk.KernelDensity(bandwidth=self.bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
            kde.fit(C0[:,feature].reshape(-1, 1))
            self.kdes_c0.append(kde)
    
            kde = sk.KernelDensity(bandwidth=self.bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
            kde.fit(C1[:,feature].reshape(-1, 1))
            self.kdes_c1.append(kde)
            
    def predict(self, X):
        """
        """
        # Start with the prior probability
        pc0 = self.P_c0
        pc1 = self.P_c1
        
        # For each feature, add the log of the conditional probability of the feature given the class
        for feature in range(self.n_features):
            pc0 += self.kdes_c0[feature].score_samples(X[:,feature].reshape((-1,1)))
            pc1 += self.kdes_c1[feature].score_samples(X[:,feature].reshape((-1,1)))
    
        # Choose the highest probability as the class of each example
        return pc1 > pc0
    
    