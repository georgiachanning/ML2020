#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv

def main():
    
    # weights = [1,2,4,6,8,10,12,14,16,18,20,22]
    weights = np.ones(12)
    
    #data preprocessing
    pre_train_features = np.genfromtxt('test_features.csv',delimiter=',', skip_header = 1)
    #np.genfromtxt('train_features.csv',delimiter=',', skip_header = 1)
    
        
    #remove nan
    train_features = []
        
    for column in pre_train_features.T:
        column_mean = np.nanmean(column)
        train_features.append(np.nan_to_num(column, nan=column_mean))
    
    train_features = np.array(train_features).T
    print(train_features)
        
    #average over features
    pid_matrices = {}
    rows = []
    all_pid_features_averaged = []
           
    previous_pid = 0
    for row in train_features:
        pid = row[0]
        if pid != previous_pid:
            pid_matrices[previous_pid] = rows
            
            pid_matrix = np.array(rows)
            pid_avg = np.average(pid_matrix, axis = 0, weights=weights)
            all_pid_features_averaged.append(pid_avg)
            
            rows.clear()
        rows.append(row)
        previous_pid = pid
        
    #for last row
    pid_matrices[previous_pid] = rows
    pid_matrix = np.array(rows)
    pid_avg = np.average(pid_matrix, axis = 0, weights=weights)
    all_pid_features_averaged.append(pid_avg)

    #write out
    with open("preprocessed_nonzero_test_features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_pid_features_averaged)
        
    return

if __name__ == '__main__':
    main()

