#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:17:31 2022

LightGBM with user-defined negative binomial loss

When using the repository, please cite our work:

@article{long2025scalable,
  title={Scalable probabilistic forecasting in retail with gradient boosted trees: A practitioner�s approach},
  author={Long, Xueying and Bui, Quang and Oktavian, Grady and Schmidt, Daniel F and Bergmeir, Christoph and Godahewa, Rakshitha and Lee, Seong Per and Zhao, Kaifeng and Condylis, Paul},
  journal={International Journal of Production Economics},
  volume={279},
  pages={109449},
  year={2025},
  publisher={Elsevier}
}

"""

import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pyreadr
import subprocess
import os
import sys
sys.path.append(".")

from scipy.optimize import minimize

from utils.optimization import negbin_likelihood
from utils.lgb_negbin_loss import neg_bin_loss, neg_bin_vaild

import utils.global_vars as global_vars


if __name__ == "__main__":
    
    # set seed
    random.seed(0)
    np.random.seed(0)
    
    ######## constants ########
    # forecast horizon
    H = 28
    
    # lags
    lag = 100
    
    # fourier terms
    fourier_periods = [7, 365.25]
    
    # lgb params: default
    lgb_params = {
        'objective': 'regression',
        'random_seed': 0,
        # 'bagging_freq': 1,
        # 'lambda_l2': 0.1,
        # 'learning_rate': 0.075,
        # 'num_leaves': 128,
        # 'min_data_in_leaf': 100,
        # 'boosting_type': 'gbdt',
        # 'force_row_wise': True,
        # 'sub_row': 0.75,
        # 'verbosity': -1,
        # 'num_iterations': 1200
    }
    
    BASE_DIR = "~/probabilistic-forecasting"
    
    ######## Data ########
    
    # load data
    sku_data = pyreadr.read_r("./data/M5_level_10.rds")
    sku_data = sku_data[None]
    sku_data = sku_data.sort_values(by=["item_id"])
    sku_id = sku_data.iloc[:, 0]
    sku_data = sku_data.iloc[:, 1:]
    
    global_vars.r_negbin = sku_data.mean(axis=1) ** 2 / (sku_data.var(axis=1) - sku_data.mean(axis=1))
    global_vars.num_sku = len(sku_data)
    
    # create training and testing sets
    if not os.path.isfile("./data/M5_level_10_embedded_matrix_fourier_terms_full.rds"):
        subprocess.call(["Rscript", "--vanilla", BASE_DIR + "/utils/negbin_loss_data_helper.R", BASE_DIR, "/data/M5_level_10.rds", "M5_level_10", "item_id", str(lag), str(H)])
    train_df = pyreadr.read_r("./data/M5_level_10_embedded_matrix_fourier_terms_full.rds")
    train_df = train_df[None]
    # get the test set
    test_df = pd.read_csv("./data/M5_level_10_final_lags.csv")
    
    # training
    lgb_train = lgb.Dataset(train_df.iloc[:, 1:], train_df.iloc[:,0])
    print("start training...")
    start = time.time()
    model = lgb.train(lgb_params, 
                lgb_train,
                fobj=neg_bin_loss,
                feval=neg_bin_vaild)
    
    pred = np.exp(model.predict(train_df.iloc[:, 1:]))
    
    
    # ######## coordinate-wise optimisation: learn r_negbin ########
    print("finding the optimal r value...")
    
    itera = 0
    while itera < 10: # add more iteration when more compute is available, the iteration is set to the same across the 3 datasets.
        itera += 1
        print("current updating iteration...", itera)
        pred = np.exp(model.predict(train_df.iloc[:, 1:]))
        bnds = ((1e-6, None),)
        result = []
        data_length = int(len(pred) / global_vars.num_sku)
        for i in range(global_vars.num_sku):
            res = minimize(
                negbin_likelihood,
                x0 = global_vars.r_negbin[i:i+1],
                args = (pred[data_length * i : data_length * (i+1)], train_df.iloc[:,0].to_list()[data_length * i : data_length * (i+1)], data_length, ),
                bounds=bnds
            )
            result.append(res.x[0])
        # print(max(abs(np.array(result) - np.array(global_vars.r_negbin))))
        if max(abs(np.array(result) - np.array(global_vars.r_negbin))) < 1e-3:
            break
        else:
            # update r and model
            # print("updating r...")
            global_vars.r_negbin = result
            model = lgb.train(lgb_params, 
                        lgb_train,
                        fobj=neg_bin_loss,
                        feval=neg_bin_vaild)
    
    # save models
    # model.save_model(filename = BASE_DIR + "/results/models/M5_level_10_lightgbm_negbin.txt")
    
    ######## loops ########
    
    pred_matrix = pd.DataFrame()
    for h in range(H):
        print('h = ', str(h))
        
        f = 1
        # combine the fourier matrix
        for k in range(1,5):
            for period in fourier_periods:
                feature_col = [np.sin(2.0 * np.pi * k * (H + 1 - h) / period)] * test_df.shape[0]
                test_df = pd.concat([test_df, pd.DataFrame(feature_col, columns = ["fourier_features_" + str(f)])], axis=1)
                f = f + 1
                feature_col = [np.cos(2.0 * np.pi * k * (H + 1 - h) / period)] * test_df.shape[0]
                test_df = pd.concat([test_df, pd.DataFrame(feature_col, columns = ["fourier_features_" + str(f)])], axis=1)
                f = f + 1
        
        # e^f is the final prediction
        y_pred = np.exp(model.predict(test_df))
        
        # save predictions
        # remove the last column (lag_100)
        test_df = test_df.iloc[:, :lag]
        test_df = test_df.iloc[:, :-1]
        test_df = pd.concat([pd.Series(y_pred), test_df], axis = 1)
        pred_matrix = pd.concat([pred_matrix, pd.Series(y_pred)], axis = 1)
    
    # renormalisation
    mean_list = pd.read_csv("./data/M5_level_10_series_means.csv")
    mean_list = mean_list.iloc[:,0]
    pred_matrix = pred_matrix.mul(mean_list, axis = 0)
    
    
    # save predictions to a file
    pred_matrix.insert(0, "item_id", sku_id)
    pred_matrix.to_csv("./results/forecasts/M5_level_10_lightgbm_negbin_lag_100.csv", index=False, header=False)
    end = time.time()
    if (end - start) > 3600: 
        duration = str((end - start) / 3600) + " hrs"
    elif (end - start) > 60:
        duration = str((end - start) / 60) + " mins"
    print("execution time", duration)
    with open("./results/execution_times/M5_level_10_lightgbm_negbin_lag_100.txt" , 'w') as f:
        f.write(duration)
        f.write("\n")
