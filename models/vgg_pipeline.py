import numpy as np

from keras.applications.vgg16 import VGG16

from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input

from keras.models import Model

from .abstract_pipeline import AbstractPipeline

class VGGPipeline(AbstractPipeline):
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.input_resize_to = (224, 224)


    def get_train_samples(self, df):
        return len(df) * 3 * 2


    def get_validation_samples(self, df):
        return len(df)


    def preprocess_image(self, image):
        image_np = np.asarray(image.resize(self.input_resize_to))
        return image_np


    def get_train_generator(self, data_folder, batch_size=64):
        return self.get_left_center_right_generator(data_folder, batch_size)


    def get_validation_generator(self, data_folder, batch_size=64):
        return self.get_center_only_generator(data_folder, batch_size)
        

    def get_model(self):
        input_layer = Input(shape=self.input_shape)
        model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)

        # freeze layers
        for layer in model.layers:
            layer.trainable = False

        layer = model.outputs[0]

        layer = Flatten()(layer)
        layer = Dropout(.2)(layer)
        layer = Dense(2000, activation='relu')(layer)
        layer = Dropout(.3)(layer)
        layer = Dense(1000, activation='relu')(layer)
        layer = Dropout(.3)(layer)
        layer = Dense(500, activation='relu')(layer)
        layer = Dropout(.4)(layer)
        layer = Dense(100, activation='relu')(layer)
        layer = Dropout(.5)(layer)
        layer = Dense(1)(layer)

        return Model(input=model.input, output=layer)