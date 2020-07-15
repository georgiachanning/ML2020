#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold,KFold
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.preprocessing import scale
import random
import csv
import pandas as pd
import sklearn.metrics as metrics


#naive random upsample by me
def scale_data(train_data, test_data):
    len_train_data = len(train_data)
    all_data = np.concatenate((train_data,test_data), axis=0)
    all_data_scaled = scale(all_data)
    scaled_train_data = all_data_scaled[0:len_train_data,:]
    scaled_test_data = all_data_scaled[len_train_data:,:]
    return scaled_train_data, scaled_test_data
    

def upsample(X, y):
    unique, counts = np.unique(y, return_counts=True)
    numP = dict(zip(unique, counts))[1]
    numN = dict(zip(unique, counts))[0]
    indices_of_P = [i for i, x in enumerate(y) if x == 1]
    while numP != numN:
        index_to_repeat = random.choice(indices_of_P)  
        X = np.vstack((X, np.reshape(X[index_to_repeat],(1,35))))
        y = np.append(y, 1)
        numP += 1
    return X, y

def main():
    
    #fixed
    k_fold = StratifiedKFold(n_splits=2)
    fold3 = KFold(n_splits=2)
    sm = SMOTE(random_state=42)

    #data read
    pids = np.genfromtxt('preprocessed_nonzero_test_features.csv',delimiter=',')[:,0]
    train_features = np.genfromtxt('preprocessed_nonzero_train_features.csv',delimiter=',')
    # train_features_weighted = np.genfromtxt('preprocessed_nonzero_train_features.csv',delimiter=',')
    train_labels = np.genfromtxt('train_labels.csv',delimiter=',', skip_header=1)
    
    # test_features_weighted = np.genfromtxt('preprocessed_weighted_test_features.csv',delimiter=',')
    test_features = np.genfromtxt('preprocessed_nonzero_test_features.csv',delimiter=',')
    
    train_features, test_features = scale_data(train_features, test_features)
    
    #subtask1--training
    reg1 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=5), learning_rate = 0.1, n_estimators = 100, random_state = 42)
    roc_auc_1 = []
    X1 = train_features[:,2:]
    y1_test = []
    for column in range(1,11):
        print(column)
        print('\n')
        y1 = train_labels[:,column]
        
        for k, (train, val) in enumerate(k_fold.split(X1, y1)):
            # X1_train, y1_train = sm.fit_resample(X1[train], y1[train])
            reg1.fit(X1[train], y1[train])
            y1_pred = reg1.predict(X1[val])
            roc_auc_1.append(roc_auc_score(y1[val], y1_pred))
            print(roc_auc_score(y1[val], y1_pred))
            
            
        #testing
        reg1.fit(X1, y1)
        X1_test = test_features[:,2:]
        y1_test.append(reg1.predict(X1_test))
            
        
    #subtask2
    y2 = train_labels[:,11]
    X2 = train_features[:,2:]
    print(11)
        
    #training
    roc_auc_2 = []
    reg2 = GradientBoostingRegressor(warm_start=False)
    # reg2 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=10), learning_rate = 0.1, n_estimators = 100, random_state = 42)
    for k, (train, val) in enumerate(k_fold.split(X2, y2)):
        X2_train, y2_train = sm.fit_resample(X2[train], y2[train])
        reg2.fit(X2_train, y2_train)
        y2_pred = reg2.predict(X2[val])
        roc_auc_2.append(roc_auc_score(y2[val], y2_pred))
        print(y2_pred)
        print(y2[val])
    print(roc_auc_2)
    
    
    #testing
    X2, y2 = sm.fit_resample(X2, y2)
    reg2.fit(X2, y2)
    X2_test = test_features[:,2:]
    y2_test = reg2.predict(X2_test)
    

    #subtask3
    reg3 = svm.SVR()
    X3 = train_features[:,2:]
    r_2 = []
    y3_test = []
    for column in range(12,16):
        print(column)
        print('\n')
        y3 = train_labels[:,column]
        
        for k, (train, val) in enumerate(fold3.split(X3, y3)):
            reg3.fit(X3[train], y3[train])
            y3_pred = reg3.predict(X3[val])
            r_2.append(r2_score(y3[val], y3_pred))
            print(r2_score(y3[val], y3_pred))
            print(y3_pred)
            print(y3[val])
        
        #testing
        reg3.fit(X3, y3)
        X3_test = test_features[:,2:]
        y3_test.append(reg3.predict(X3_test))
    
    
    #writeout
    header = "pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate"
    formatted_data = np.vstack((pids,y1_test[0], y1_test[1], y1_test[2],y1_test[3],y1_test[4],y1_test[5],y1_test[6],y1_test[7],y1_test[8],y1_test[9],y2_test,y3_test[0],y3_test[1],y3_test[2],y3_test[3]))
    
    np.savetxt('submission.csv', X = np.transpose(formatted_data), header=header, fmt='%.3f', delimiter=',')
    

    return

if __name__ == '__main__':
    main()