from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import ELU

from scipy.misc import imread, imresize

from .abstract_pipeline import AbstractPipeline

from keras.regularizers import l2, activity_l2

import numpy as np

class CommaAiPipeline(AbstractPipeline):

    def __init__(self):
        self.input_shape = (160, 320, 3)


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
        return image_np
        

    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1.,
                    input_shape=self.input_shape))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        return model
