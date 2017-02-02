# See models/nvidia_pipeline.py for the model architecture

import sys
import cv2
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from models.nvidia_pipeline import NvidiaPipeLine
from models.vgg_pipeline import VGGPipeline
from models.small_image_pipeline import SmallImagePipeline
from models.comma_ai_pipeline import CommaAiPipeline

from PIL import Image
from utils import get_driving_log_dataframe
from utils import get_callbacks
from keras.models import load_model

# Instead of implementing here, the model is in the
# models/ directory, because this allows to quickly
# switch between different pipelines
pipeline = NvidiaPipeLine()

BATCH_SIZE = 32
EPOCHS = 2

# this function is also used in drive.py,
# this way when switching the pipeline the 
# preprocessing for driving is also changed
# appropriately
def preprocess(image):
    return pipeline.preprocess_image(image)


def train(data_folder, validation_folder, restart_model_path=None):
    if restart_model_path:
        model = load_model(restart_model_path)
        print("Using existing model")
    else:
        model = pipeline.get_model()
        model.compile("adam", "mse")
        print("Using new model")

    samples = pipeline.get_train_samples(get_driving_log_dataframe(data_folder))

    train_generator = pipeline.get_train_generator(data_folder, batch_size=BATCH_SIZE)

    model.summary()

    image_generator = train_generator
    validation_generator = pipeline.get_validation_generator(validation_folder, batch_size=BATCH_SIZE)
    nb_val_samples = pipeline.get_validation_samples(get_driving_log_dataframe(validation_folder))

    # callbacks that save weights after each epoch
    callbacks_list = get_callbacks()

    model.fit_generator(image_generator, 
                        samples_per_epoch=samples, 
                        nb_epoch=EPOCHS,
                       callbacks=callbacks_list,
                       validation_data=validation_generator,
                       nb_val_samples=nb_val_samples)

    # save everything for possible finetuning in the future
    model.save('model-compiled.h5')

    json_string = model.to_json()
    with open('model.json', 'w') as model_json_file:
        model_json_file.write(json_string)

    model.save_weights('model.h5')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python model.py train_folder valid_folder [exising_model_to_finetune]')
    elif len(sys.argv) < 4:
        train(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
