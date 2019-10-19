#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:14:55 2019

@author: af.rosa
"""

import sklearn.neighbors as sk
import math

class naiive_bayes_classifier:
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.P_c0 = 0
        self.P_c1 = 0
        self.kdes_c0 = []
        self.kdes_c1 = []
        self.n_features = 0
        
    def fit(self, train_data):
        """
        """
        C0 = train_data[train_data[:,-1] == 0.0]
        C1 = train_data[train_data[:,-1] == 1.0]

        self.P_c0 = math.log(C0.shape[0] / train_data.shape[0])
        self.P_c1 = math.log(C1.shape[0] / train_data.shape[0])
        
        self.n_features = train_data.shape[1] - 1
        
        for feature in range(self.n_features):
            kde = sk.KernelDensity(bandwidth=self.bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
            kde.fit(C0[:,feature].reshape(-1, 1))
            self.kdes_c0.append(kde)
    
            kde = sk.KernelDensity(bandwidth=self.bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
            kde.fit(C1[:,feature].reshape(-1, 1))
            self.kdes_c1.append(kde)
            
    def predict(self, examples):
        """
        """
        pc0 = self.P_c0
        pc1 = self.P_c1
        for feature in range(self.n_features):
            pc0 += self.kdes_c0[feature].score_samples(examples[:,feature].reshape((-1,1)))
            pc1 += self.kdes_c1[feature].score_samples(examples[:,feature].reshape((-1,1)))
    
        return pc1 > pc0