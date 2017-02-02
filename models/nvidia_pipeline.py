from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import ELU

from .abstract_pipeline import AbstractPipeline

from keras.regularizers import l2, activity_l2

import numpy as np

class NvidiaPipeLine(AbstractPipeline):

    def __init__(self):
        self.input_shape = (64, 64, 3)
        self.input_resize_to = (64, 64)


    def get_train_samples(self, df):
        return len(df) * 4


    def get_validation_samples(self, df):
        return len(df)


    def preprocess_image(self, image):
        image = image.convert('YCbCr')
        image_np = np.asarray(image)
        image_np = self.crop(image_np)
        image_np = self.resize(image_np, self.input_resize_to)

        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential()

        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

        # starts with five convolutional and maxpooling layers
        model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Flatten())

        # Next, five fully connected layers
        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(50))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(1))

        return model
      
