from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda

from scipy.misc import imread, imresize

from .abstract_pipeline import AbstractPipeline

from keras.regularizers import l2, activity_l2

import numpy as np

class SmallImagePipeline(AbstractPipeline):

    def __init__(self):
        self.input_shape = (16, 32, 1)
        self.input_resize_to = (32, 16)


    def get_train_samples(self, df):
        return len(df) * 3 * 2


    def rgb2gray(self, img):
        grayed = np.mean(img, axis=2, keepdims=True)
        return grayed

    def normalize(self, img):
        return img / (255.0 / 2) - 1


    def get_validation_samples(self, df):
        return len(df)
    

    def preprocess_image(self, image):
        image_np = np.asarray(image)
        image_np = imresize(image_np, (32, 16, 3))
        image_np = self.rgb2gray(image_np)
        image_np = self.normalize(image_np)
        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential([
            Conv2D(32, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
            Conv2D(64, 3, 3, border_mode='same', activation='relu'),
            Dropout(0.5),
            Conv2D(128, 3, 3, border_mode='same', activation='relu'),
            Conv2D(256, 3, 3, border_mode='same', activation='relu'),
            Dropout(0.5),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, name='output', activation='tanh'),
        ])
        return model
