from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda

from .abstract_pipeline import AbstractPipeline

from keras.regularizers import l2, activity_l2

import numpy as np

class SmallImagePipeline(AbstractPipeline):

    def __init__(self):
        self.input_shape = (16, 32, 3)
        self.input_resize_to = (32, 16)


    def get_train_samples(self, df):
        return len(df) * 3 * 2


    def get_validation_samples(self, df):
        return len(df)


    def preprocess_image(self, image):
        #image = image.convert('L')
        image_np = np.asarray(image.resize(self.input_resize_to))
#        image_np = np.asarray(image)
#        image_np = np.reshape(image_np, (16, 32, 1))
        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(16, 3, 3, input_shape=self.input_shape, border_mode='same', activation='relu'))
        model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))
        return model
