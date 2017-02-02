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

# Slightly modified Nvidia model
# to be less computationally expensive
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
class NvidiaPipeLine(AbstractPipeline):

    def __init__(self):
        # in the original paper, the images were with size (66, 200)
        self.input_shape = (64, 64, 3)
        self.input_resize_to = (64, 64)


    def get_train_samples(self, df):
        return len(df) * 4


    def get_validation_samples(self, df):
        return len(df)


    def preprocess_image(self, image):
        # convert to YUV
        image = image.convert('YCbCr')
        image_np = np.asarray(image)
        # crop the top of the image, which does not add
        # useful information for the predictions
        image_np = self.crop(image_np)
        image_np = self.resize(image_np, self.input_resize_to)

        return image_np


    # there are two types of generators in abstract_pipeline:
    # one for dataframe with only center images, and one with 
    # left, center, right images. This is so that we can use
    # both the keyboard as well as beta simulator
    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        # Note that this follows the Nvidia paper, but has added
        # MaxPooling layers after Convolutional layers, which are not
        # present in the Nvidia paper

        model = Sequential()

        # normalization layer
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))

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

        model.add(Dense(1164))
        model.add(Activation('relu'))

        # Dropout to prevent overfitting
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
      
