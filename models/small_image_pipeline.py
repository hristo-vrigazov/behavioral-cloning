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

class SmallImagePipeline(AbstractPipeline):

    def __init__(self):
        self.input_shape = (64, 42, 3)
        self.input_resize_to = (42, 64)


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
        image_np = self.crop(image_np)
        image_np = self.resize(image_np, self.input_resize_to)

        # 1 in 4 images will be with augmented brightness
        toss = np.random.random()
        if toss <= .25:
          image_np = self.augment_brightness_camera_images(image_np)

        toss = np.random.random()
        if toss <= .25:
          image_np = self.add_random_shadow(image_np)

#        image_np = self.rgb2gray(image_np)
        image_np = self.normalize(image_np)
        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        model = Sequential()

        model.add(Convolution2D(16, 5, 5, input_shape=self.input_shape, subsample=(2, 2), border_mode="same"))
        model.add(ELU())

        model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode="valid"))
        model.add(ELU())
        model.add(Dropout(.5))
        model.add(MaxPooling2D((2, 2), border_mode='valid'))

        model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
        model.add(ELU())
        model.add(Dropout(.5))

        model.add(Flatten())

        model.add(Dense(1024))
        model.add(Dropout(.3))
        model.add(ELU())

    # layer 5
        model.add(Dense(512))
        model.add(ELU())

    # Finally a single output, since this is a regression problem
        model.add(Dense(1))
        return model
