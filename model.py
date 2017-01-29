import sys
import cv2
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from models.nvidia_pipeline import NvidiaPipeLine
from models.vgg_pipeline import VGGPipeline
from models.small_image_pipeline import SmallImagePipeline

from PIL import Image
from utils import get_driving_log_dataframe
from utils import get_callbacks
from keras.models import load_model
# Instead of implementing here, the model is in the
# models/ directory, because this allows to quickly
# switch between different pipelines
pipeline = SmallImagePipeline()

BATCH_SIZE = 32
EPOCHS = 3

def preprocess(image):
    return pipeline.preprocess_image(image)


def get_preprocessed_dataframe(data_folder, batch_size=64):
    angles_df = pd.read_csv('{}/angles.csv'.format(data_folder), header=None)
    angles_df = angles_df.reindex(np.random.permutation(angles_df.index))

    number_of_examples = len(angles_df)

    while True:
        image_series = angles_df[0]
        steering_series = angles_df[1]
        for offset in range(0, number_of_examples, batch_size):
            X_train = []
            y_train = []
            weights = []

            end_of_batch = min(number_of_examples, offset + batch_size)

            for j in range(offset, end_of_batch):
               image_filename = image_series[j].lstrip().rstrip()
                    
               image = Image.open('{0}/{1}'.format(data_folder, image_filename))
               image_np = preprocess(image)
               label = steering_series[j]
                    
               X_train.append(image_np)
               y_train.append(label)
               weights.append(1)
                    
               flipped_image = np.fliplr(image_np)
               flipped_label = -label
                    
               X_train.append(flipped_image) 
               y_train.append(flipped_label)
               weights.append(1)
                
                    
               X_train, y_train, weights = shuffle(X_train, y_train, weights)
               yield np.array(X_train), np.array(y_train), np.array(weights)


def train(data_folder, validation_folder, restart_model_path=None, train_generator=None):
    if restart_model_path:
        model = load_model(restart_model_path)
        print("Using existing model")
    else:
        model = pipeline.get_model()
        model.compile("adam", "mse")
        print("Using new model")

    samples = pipeline.get_train_samples(get_driving_log_dataframe(data_folder))

    if not train_generator:
        train_generator = pipeline.get_train_generator(data_folder, batch_size=BATCH_SIZE)
    else:
        samples = 96126

    model.summary()

    image_generator = train_generator
    validation_generator = pipeline.get_validation_generator(validation_folder, batch_size=BATCH_SIZE)
    nb_val_samples = pipeline.get_validation_samples(get_driving_log_dataframe(validation_folder))
    callbacks_list = get_callbacks()

    model.fit_generator(image_generator, 
                        samples_per_epoch=samples, 
                        nb_epoch=EPOCHS,
                       callbacks=callbacks_list,
                       validation_data=validation_generator,
                       nb_val_samples=nb_val_samples)

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
#        train(sys.argv[1], sys.argv[2], train_generator=get_preprocessed_dataframe('all_preprocessed', BATCH_SIZE))
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
