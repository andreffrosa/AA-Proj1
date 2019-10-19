#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:48:00 2019

@author: andreffrosa
"""

C1 = train_data[train_data[:,-1] == 1.0]
C0 = train_data[train_data[:,-1] == 0.0]

import math

P_c0 = math.log(C0.shape[0] / train_data.shape[0])
P_c1 = math.log(C1.shape[0] / train_data.shape[0])

n_features = train_data.shape[1] - 1

import sklearn.neighbors as sk

kdes_c0 = []
kdes_c1 = []
for feature in range(n_features):
    kde = sk.KernelDensity(bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean')
    kde.fit(C0[:,feature].reshape(-1, 1))
    kdes_c0.append(kde)
    
    kde = sk.KernelDensity(bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean')
    kde.fit(C1[:,feature].reshape(-1, 1))
    kdes_c1.append(kde)


def predict(example):
    pc0 = P_c0
    pc1 = P_c1
    for feature in range(n_features):
        pc0 += kdes_c0[feature].score(example[feature].reshape((-1,1)))
        pc1 += kdes_c1[feature].score(example[feature].reshape((-1,1)))
    
    if pc0 > pc1:
        return 0
    else:
        return 1
    
def predict2(examples):
    pc0 = P_c0
    pc1 = P_c1
    for feature in range(n_features):
        pc0 += kdes_c0[feature].score_samples(examples[:,feature].reshape((-1,1)))
        pc1 += kdes_c1[feature].score_samples(examples[:,feature].reshape((-1,1)))
    
    return pc1 > pc0