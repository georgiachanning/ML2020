#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
import csv
from sklearn.preprocessing import scale, StandardScaler
from imblearn.over_sampling import SMOTE


def scale_data(train_data, test_data):
    len_train_data = len(train_data)
    all_data = np.concatenate((train_data,test_data), axis=0)
    all_data_scaled = scale(all_data)
    scaled_train_data = all_data_scaled[0:len_train_data,:]
    scaled_test_data = all_data_scaled[len_train_data:,:]
    return scaled_train_data, scaled_test_data


def main():
    
    #fixed 
    skf = StratifiedKFold(5)
    os = SMOTE(0.15)
    scaler = StandardScaler()
    
    #load data and preprocessing
    train_csv = csv.reader(open('train.csv'))
    next(train_csv)
    X = []
    y = []
    for row in train_csv:
        for letter in list(row[0]):
            X.append(ord(letter))
        y.append(eval(row[1]))
    
    X = np.reshape(np.array(X), (112000,4))
    y = np.array(y)
    
    test_csv = csv.reader(open('test.csv'))
    next(test_csv)
    X_test = []
    for row in test_csv:
        for letter in list(row[0]):
            X_test.append(ord(letter))
    X_test = np.reshape(np.array(X_test), (48000,4))
    
    # X, X_test = scale_data(X, X_test)
    
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    #training    
    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), algorithm="SAMME", random_state=7, n_estimators=100)
    # clf = HistGradientBoostingClassifier(max_leaf_nodes=None, l2_regularization=2.0)
    clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=0.01, max_iter=10000, warm_start=False, hidden_layer_sizes=(8,8,8,8,), random_state=333)
    
    '''for k, (train, val) in enumerate(skf.split(X, y)):
        X_train, y_train = os.fit_resample(X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[val])
        print(confusion_matrix(y[val], y_pred))
        
        
        print(roc_auc_score(y[val], y_pred))
        print(precision_score(y[val], y_pred))
        print(recall_score(y[val], y_pred))
        print(accuracy_score(y[val], y_pred))
        print(f1_score(y[val], y_pred))'''
        
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, y_train = os.fit_resample(X_train, y_train)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    print(confusion_matrix(y_val, y_pred))
    print(roc_auc_score(y_val, y_pred))
    print(precision_score(y_val, y_pred))
    print(recall_score(y_val, y_pred))
    print(accuracy_score(y_val, y_pred))
    print(f1_score(y_val, y_pred))
    
        
    
    y_pred = clf.predict(X_test)
    with open('submission.csv', 'w') as out:
        wr = csv.writer(out, delimiter='\n')
        wr.writerow(y_pred)
        
    return


if __name__ == '__main__':
    main()


