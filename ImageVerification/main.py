import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
import random as python_random
from sklearn.decomposition import PCA

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(7)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(7)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(78)


from datagen import DataGenerator
from datagen_feature import DataGenerator_features
import models

'''policy = keras.mixed_precision.experimental.Policy('mixed_float16')
keras.mixed_precision.experimental.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
'''
# A solution for a problem (found online)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# tf.config.experimental_run_functions_eagerly(False)

# Load the data triplets
train = np.loadtxt('train_triplets.txt').astype(int)
test = np.loadtxt('test_triplets.txt').astype(int)
labels = np.ones((train.shape[0]))

# in order to create the list of paths
n_images = 10000
path_list = []
for n in range(n_images):
    
    src = f'{n:05}.jpg'
    path_list.append(src)

# Parameters
params = {'dim': (224,224),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True,
          'image_path': 'food/',
          'pred': False}

params_test = {'dim': (224,224),
          'batch_size': 8,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False,
          'image_path': 'food/',
          'pred': True}

# Datasets
partition_tr = np.arange(train.shape[0])
partition_test = np.arange(test.shape[0])

file_names_tr = np.asarray(path_list)[train]
file_names_test = np.asarray(path_list)[test]

# Generators
train_generator = DataGenerator(partition_tr,file_names_tr,labels, **params)
y_dummy = np.zeros(test.shape[0])
test_generator = DataGenerator(partition_test,file_names_test,y_dummy, **params_test)

print("Generating the model...")

input_shape = (224,224,3)

pre_model = tf.keras.applications.NASNetMobile(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='avg',
    classes=1000,
)

# Freeze the model
index = None
for idx, layer in enumerate(pre_model.layers):
    if layer.name == 'normal_right4_9':
        index = idx
        break

for idx, layer in enumerate(pre_model.layers):
    if idx <= index:
        layer.trainable = False

inputA = keras.Input(shape=input_shape)
inputB = keras.Input(shape=input_shape)
inputC = keras.Input(shape=input_shape)

procA = pre_model(inputA)
procB = pre_model(inputB)
procC = pre_model(inputC)

x = keras.layers.Lambda(models.euclidean_distance,
                output_shape=models.eucl_dist_output_shape)([procA, procB])
y = keras.layers.Lambda(models.euclidean_distance,
                output_shape=models.eucl_dist_output_shape)([procA, procC])

conc = keras.layers.concatenate([x,y])

z = keras.activations.softmax(conc,axis=-1)

model = keras.models.Model([inputA,inputB,inputC], z)

model.summary()

adam_optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss=models.t_loss,
              optimizer=adam_optim,
              metrics=['acc'])

print("Model Generated!")

# Train model on dataset
history = model.fit(x=train_generator,
                    epochs = 1, verbose = 1)

params_features = {'dim': (224,224),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False,
          'image_path': 'food/',
          'pred': True}

# Datasets
partition_features = np.arange(10000)

file_names_features = np.asarray(path_list)

y_features_dummy = np.zeros(partition_features.shape[0])
features_generator = DataGenerator_features(partition_features,file_names_features,y_features_dummy, **params_features)

prediction = pre_model.predict(features_generator,verbose = 1)
model.save_weights('model3.h5')
del model
del pre_model
data = np.copy(prediction)

# PCA for dimensionality reduction against noise
pca = PCA(n_components=4)
pca.fit(data[:5000])
data = pca.transform(data)

ind = train.shape[0]//2
col1 = train[ind:,1].copy()
col2 = train[ind:,2].copy()
train[ind:,1] = col2.copy()
train[ind:,2] = col1.copy()
labels[ind:] = 0

partition_tr = np.arange(train.shape[0])
file_names_tr = np.asarray(path_list)[train]
# Generators
train_generator = DataGenerator(partition_tr,file_names_tr,labels, **params)

print("Generating the model...")

input_shape = (12,)

inputy = keras.Input(shape=input_shape)
y = keras.layers.Dense(4000,
                activation='relu')(inputy)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(1, activation='sigmoid')(y)

model_classif = keras.models.Model(inputy, y)
model_classif.summary()

adam_optim = keras.optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999)
model_classif.compile(loss='binary_crossentropy',
              optimizer=adam_optim,
              metrics=['acc'])

print("Model Generated!")

datax = np.concatenate((data[train][:,0], data[train][:,1], data[train][:,2]),axis=1)

# Train model on dataset
history = model_classif.fit(x=datax,y =labels,
                    shuffle = True, epochs = 12, verbose = 1)

# prediction on test data
datay = np.concatenate((data[test][:,0],data[test][:,1],data[test][:,2]),axis=1)

prediction = model_classif.predict(datay,verbose = 1)
print(prediction)

np.savetxt('prediction_prob3.txt',prediction)
np.savetxt('prediction3.txt', np.round(prediction),fmt = '%.d')
