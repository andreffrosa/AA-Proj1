#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:36:49 2019

@author: andreffrosa
"""

import numpy as np

#
    
c_values = np.linspace(0.01,100,30)
def calc_C(index):
    return c_values[index]

for i in range(30):
    print(calc_C(i))