#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:33:27 2022

"""

import numpy as np
import utils.global_vars as global_vars


def neg_bin_loss(y_pred, dtrain):
    y_true = dtrain.get_label()

    r = np.repeat(global_vars.r_negbin, int(len(y_true) / global_vars.num_sku))
    
    grad = np.exp(y_pred) * (r + y_true) / (np.exp(y_pred) + r) - y_true
    hess = np.exp(y_pred) * (r ** 2 + y_true * r) / (np.exp(y_pred) + r) ** 2
 
    return grad, hess

def neg_bin_vaild(y_pred, dtrain):
    y_true = dtrain.get_label()

    r = np.repeat(global_vars.r_negbin, int(len(y_true) / global_vars.num_sku))
    
    loss = -1 * y_true * y_pred + y_true * np.log(np.exp(y_pred) + r) + r * np.log(np.exp(y_pred) + r)
    
    return "neg_bin_vaild", loss.mean(), False
