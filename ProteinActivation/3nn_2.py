#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from sklearn.metrics import roc_curve
from tensorflow.keras import layers
import tensorflow_addons as tfa

tf.random.set_seed(54321)

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))

def df_to_dataset(dataframe, labels, shuffle=True, batch_size=16):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def test_to_dataset(dataframe, shuffle=False, batch_size=16):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def main():
    #load data and preprocessing
    train_data = pd.read_csv('train.csv')
    train, val = train_test_split(train_data, test_size=0.2, random_state=123)
    test_data = pd.read_csv('test.csv')
        
    #define columns for Dataset API
    CATEGORICAL_COLUMNS = ['site1','site2','site3','site4']
    
    #get labels
    train_labels = train.pop('Active')
    val_labels = val.pop('Active')
    all_train_labels = train_data.pop('Active')
    
    
    #extracting amino acids
    all_train_features = pd.DataFrame(train_data["Sequence"].str.split('', expand=True).values, columns=['blank1', 'site1','site2','site3','site4', 'blank2'])
    all_train_features = all_train_features.filter(['site1','site2','site3','site4'], axis=1)
    
    train_features = pd.DataFrame(train["Sequence"].str.split('', expand=True).values, columns=['blank1', 'site1','site2','site3','site4', 'blank2'])
    train_features = train_features.filter(['site1','site2','site3','site4'], axis=1)
    
    val_features = pd.DataFrame(val["Sequence"].str.split('', expand=True).values, columns=['blank1', 'site1','site2','site3','site4', 'blank2'])
    val_features = val_features.filter(['site1','site2','site3','site4'], axis=1)
    
    test_features = pd.DataFrame(test_data["Sequence"].str.split('', expand=True).values, columns=['blank1', 'site1','site2','site3','site4', 'blank2'])
    test_features = test_features.filter(['site1','site2','site3','site4'], axis=1)
    
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
      vocabulary = train_features[feature_name].dropna().unique()
      feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
    
    all_train_input_fn = df_to_dataset(all_train_features, all_train_labels)
    train_input_fn = df_to_dataset(train_features, train_labels)
    eval_input_fn = df_to_dataset(val_features, val_labels, shuffle=False)
    test_input_fn = test_to_dataset(test_features, shuffle=False)
    
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(512, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    
    model.fit(train_input_fn, validation_data=eval_input_fn, epochs=15, verbose=True, class_weight={0:1,1:2})
    predictions= (model.predict(test_input_fn) > 0.5).astype("int32")
        
    #write out predictions
    submission = pd.DataFrame(predictions)
    submission.to_csv('submission.csv', header=False, index=False, index_label=False)
    

    
    return

if __name__ == "__main__":
    main()