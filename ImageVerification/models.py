import numpy as np
import tensorflow as tf
from tensorflow import keras

def t_loss(y_true,y_pred):
    '''Loss function found in https://arxiv.org/abs/1412.6622
    '''

    return 2*keras.backend.square(y_pred[:,0])



def triplet_loss(layer):
    def loss(y_true,y_pred):

        bce2 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        probs = keras.backend.softmax(layer,axis=1)
        
        return bce2
   
    # Return a function
    return loss

def triplet_loss1(layer):
    def loss(y_true,y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
            
        print(layer)
        total_lenght = layer.shape.as_list()[-1]

        anchor = layer[:,0:int(total_lenght*1/3)]
        positive = layer[:,int(total_lenght*1/3):int(total_lenght*2/3)]
        negative = layer[:,int(total_lenght*2/3):int(total_lenght*3/3)]

        print(anchor)
        print(anchor - positive)

        # distance between the anchor and the positive
        pos_dist = keras.backend.sqrt(keras.backend.sum(keras.backend.square(anchor-positive),axis=1, keepdims = True))

        # distance between the anchor and the negative
        neg_dist = keras.backend.sqrt(keras.backend.sum(keras.backend.square(anchor-negative),axis=1, keepdims = True))
        print(keras.backend.shape(pos_dist))
        print(pos_dist)
        conc = keras.backend.concatenate((pos_dist,neg_dist),axis = 1)
        print(conc)
        probs = keras.backend.softmax(conc,axis=1)
        print(probs)
        print(probs[:,0])
        bce2 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return keras.backend.square(probs[:,0]) + bce2
   
    # Return a function
    return loss

def triplet_loss0(layer, alpha = 0.4):
    def loss(y_true, y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        # print('y_pred.shape = ',y_pred)
        
        total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12
        
        anchor = y_pred[:,0:int(total_lenght*1/3)]
        positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
        negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

        # distance between the anchor and the positive
        pos_dist = keras.backend.sum(keras.backend.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = keras.backend.sum(keras.backend.square(anchor-negative),axis=1)
        
        
        #if y_true[0] == 1:
        basic_loss = pos_dist - neg_dist+alpha
        return basic_loss
        #else:
            #basic_loss = neg_dist - pos_dist+alpha

        # compute loss
        #loss = keras.backend.maximum(basic_loss,0.0)
        #print(pos_dist,neg_dist,basic_loss)
        
    return loss

# Not in use
def my_metric_fn(y_true, y_pred):
    
    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = keras.backend.sum(keras.backend.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = keras.backend.sum(keras.backend.square(anchor-negative),axis=1)
    
    
    #if y_true[0] == 1:
    basic_loss = pos_dist - neg_dist+alpha
    
    
    return tf.reduce_mean(squared_difference, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    sum_square = keras.backend.sum(keras.backend.square(x - y), axis=1, keepdims=True)
    return keras.backend.sqrt(sum_square)
    #return keras.backend.sqrt(keras.backend.maximum(sum_square, keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def NasNet(input_shape = (224,224,3)):
    
    pre_model = tf.keras.applications.NASNetMobile(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='avg',
    classes=1000, # not important as include_top which includes the top fully connected layer is False
    )

    index = None
    for idx, layer in enumerate(pre_model.layers):
        if layer.name == 'normal_right4_9':
            index = idx
            break

    for idx, layer in enumerate(pre_model.layers):
        if idx <= index:
            layer.trainable = False

    inputA = keras.Input(shape=(224,224,3))
    inputB = keras.Input(shape=(224,224,3))
    inputC = keras.Input(shape=(224,224,3))

    procA = pre_model(inputA)
    procB = pre_model(inputB)
    procC = pre_model(inputC)
    
    x = keras.layers.concatenate([procA - procB, procA - procC])
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = keras.layers.Dense(2, activation='softmax')(x)

    model = keras.models.Model([inputA,inputB,inputC], x)
    
    return model


def ResNet(input_shape = (224,224,3), 
             dropoutRate = 0.5):
    
    
    pre_model = tf.keras.applications.ResNet50(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='avg',
    classes=1000, # not important as include_top which includes the top fully connected layer is False
    )

    for layer in pre_model.layers:
        layer.trainable = False

    inputA = keras.Input(shape=input_shape)
    inputB = keras.Input(shape=input_shape)
    inputC = keras.Input(shape=input_shape)

    procA = pre_model(inputA)
    procB = pre_model(inputB)
    procC = pre_model(inputC)


    x = keras.layers.concatenate([procA - procB, procA - procC])
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(dropoutRate)(x)
    # Add a final sigmoid layer for classification
    x = keras.layers.Dense(2, activation='sigmoid')(x)

    model = keras.models.Model([inputA,inputB,inputC], x)
    
    return model

def ownNet(nb_classes = 2, input_shape = (224,224,3)):
    
    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(2,2),
                 activation='relu')(input)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(2,2),
                 activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    
    #x = keras.layers.Dense(8,activation='relu')(x)
    #x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(2,activation='softmax')(x)

    pre_model = keras.models.Model(input, x)
    
    inputA = keras.Input(shape=input_shape)
    inputB = keras.Input(shape=input_shape)
    inputC = keras.Input(shape=input_shape)

    procA = pre_model(inputA)
    procB = pre_model(inputB)
    procC = pre_model(inputC)


    x = keras.layers.concatenate([procA, procB, procC])

    model = keras.models.Model([inputA,inputB,inputC], x)
    
    return model

def DifferentNasNet(input_shape = (224,224,3), dropoutRate = 0.5):
    
    pre_model = tf.keras.applications.NASNetMobile(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='avg',
    classes=1000, # not important as include_top which includes the top fully connected layer is False
    )

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


    x = keras.layers.Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([procA, procB])
    y = keras.layers.Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([procA, procC])

    conc = keras.layers.Concatenate(axis=1)([x,y])
    
    x = keras.activations.softmax(-conc,axis=-1)

    model = keras.models.Model([inputA,inputB,inputC], x)
    
    return model

def ownNet2(nb_classes = 2, input_shape = (224,224,3)):
    
    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(2,2),
                 activation='relu')(input)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(2,2),
                 activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    
    #x = keras.layers.Dense(8,activation='relu')(x)
    #x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(2,activation='softmax')(x)

    pre_model = keras.models.Model(input, x)
    
    inputA = keras.Input(shape=input_shape)
    inputB = keras.Input(shape=input_shape)
    inputC = keras.Input(shape=input_shape)

    procA = pre_model(inputA)
    procB = pre_model(inputB)
    procC = pre_model(inputC)


    x = keras.layers.Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([procA, procB])
    y = keras.layers.Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([procA, procC])

    conc = keras.layers.Concatenate(axis=1)([x,y])
    #x = keras.layers.Dense(1024, activation='relu')(x)
    #x = keras.layers.Dropout(dropoutRate)(x)
    # Add a final sigmoid layer for classification
    x = keras.activations.softmax(conc,axis=-1)
    
    model = keras.models.Model([inputA,inputB,inputC], x)
    
    return model
