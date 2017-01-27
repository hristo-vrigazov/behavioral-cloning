from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda

from .abstract_pipeline import AbstractPipeline

from keras.regularizers import l2, activity_l2

import numpy as np

class NvidiaPipeLine(AbstractPipeline):

    def __init__(self):
        self.input_shape = (66, 200, 3)
        self.input_resize_to = (200, 66)


    def get_train_samples(self, df):
        return len(df) * 3 * 2


    def get_validation_samples(self, df):
        return len(df)


    def preprocess_image(self, image):
        image = image.convert('YCbCr')
        image_np = np.asarray(image.resize(self.input_resize_to))
        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential()

        regularization_coef = 10e-7

        model.add(Lambda(lambda x: x/255.0,
                    input_shape=self.input_shape))
        model.add(Convolution2D(24, 
                                5, 5,
                               subsample=(2, 2),
                               init='he_normal',
                               W_regularizer=l2(regularization_coef), activity_regularizer=activity_l2(regularization_coef)))
        model.add(BatchNormalization())
        model.add(Convolution2D(36, 
                                5, 5,
                               subsample=(2, 2),
                               init='he_normal',
                               W_regularizer=l2(regularization_coef), activity_regularizer=activity_l2(regularization_coef)))
        model.add(BatchNormalization())
        model.add(Convolution2D(48, 
                                5, 5,
                               subsample=(2, 2),
                               init='he_normal',
                               W_regularizer=l2(regularization_coef), activity_regularizer=activity_l2(regularization_coef)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Convolution2D(64,
                               3, 3,
                               init='he_normal',
                               W_regularizer=l2(regularization_coef), activity_regularizer=activity_l2(regularization_coef)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Convolution2D(64,
                               3, 3,
                               init='he_normal',
                               W_regularizer=l2(regularization_coef), activity_regularizer=activity_l2(regularization_coef)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', init='he_normal'))
        model.add(Dense(50, activation='relu', init='he_normal'))
        model.add(Dense(10, activation='relu', init='he_normal'))
        model.add(Dense(1, init='he_normal'))

        return model
        
   