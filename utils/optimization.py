#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:31:34 2022

"""

import math
# math.log(0) will report an error, while np.log(0) only gives a warning
from scipy.special import loggamma
import numpy as np

# math.log does not allow vectorising
def negbin_likelihood(r, y_pred, y_true, data_length):
    likelihood =  np.sum(- loggamma(y_true + np.repeat(r, data_length)) + np.repeat(loggamma(r), data_length) + loggamma(y_true + np.repeat(1, data_length)) - np.repeat(r, data_length) * math.log(r) + r * np.log(y_pred + np.repeat(r, data_length)) - y_true * np.log(y_pred) + y_true * np.log(y_pred + np.repeat(r, data_length)) )  
    return likelihood