# modified from
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# and
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, file_names, labels, image_path, 
                 batch_size=16, dim=(224,224),
                 n_channels=3, n_classes=2, shuffle=True,pred = False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.file_names = file_names # a list of lists for multi input
        self.labels = labels 
        self.image_path = image_path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.pred = pred

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        # np.floor(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        
        if self.pred == False:
            X,y = self.__data_generate(list_IDs_temp)
            return X,y
        else:
            X = self.__data_generate(list_IDs_temp)
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generate(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        Xs = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = [np.copy(Xs),np.copy(Xs),np.copy(Xs)]
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[0][i] = self.__load_image(self.image_path + self.file_names[ID,0])
            X[1][i] = self.__load_image(self.image_path + self.file_names[ID,1])
            X[2][i] = self.__load_image(self.image_path + self.file_names[ID,2])
            
            if self.pred == False:
                y[i] = self.labels[ID]
                
        if self.pred == False:
            return X, y
        else:
            return X

        # to_categorical(y, num_classes=self.n_classes)


    def __load_image(self, image_path):
        """Load image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = image.load_img(path = image_path,grayscale=False, color_mode="rgb", \
                                            target_size=self.dim, interpolation="nearest")
        img = image.img_to_array(img).astype(np.float16)/255
        return img