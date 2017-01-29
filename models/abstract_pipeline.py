import pandas as pd
import numpy as np
import cv2
import math

from PIL import Image
from sklearn.utils import shuffle

class AbstractPipeline(object):

    def get_model(self):
        raise NotImplementedError

    def preprocess_image(self, image):
        raise NotImplementedError

    def get_train_generator(self, data_folder, batch_size=64):
        raise NotImplementedError

    def get_validation_generator(self, data_folder, batch_size=64):
        raise NotImplementedError

    def get_train_samples(self, df):
        raise NotImplementedError

    def get_validation_samples(self, df):
        raise NotImplementedError

    def get_weight(self, label):
        return 1 #math.exp(abs(label) * 2)

    def path_driving_log(self, data_folder):
        return '{}/driving_log.csv'.format(data_folder)


    def get_driving_log_dataframe(self, data_folder):
        driving_log_df = pd.read_csv(self.path_driving_log(data_folder))
        return driving_log_df

    # generator for dataframes that have left, center and right
    # as opposed to
    def get_left_center_right_generator(self, data_folder, batch_size=64):
        driving_log_df = self.get_driving_log_dataframe(data_folder)
        driving_log_df = driving_log_df.reindex(np.random.permutation(driving_log_df.index))
        number_of_examples = len(driving_log_df)
        image_columns = ['center', 'left', 'right']
        
        X_train = []
        y_train = []
        weights = []
        index_in_batch = 0
        batch_number = 0
        
        angle_offset = 0.2
        
        while True:
            for image_column in image_columns:
                image_series = driving_log_df[image_column]
                steering_series = driving_log_df['steering']
                for offset in range(0, number_of_examples, batch_size):
                    X_train = []
                    y_train = []
                    weights = []

                    end_of_batch = min(number_of_examples, offset + batch_size)

                    for j in range(offset, end_of_batch):
                        image_filename = image_series[j].lstrip().rstrip()
                        
                        image = Image.open('{0}/{1}'.format(data_folder, image_filename))
                        image_np = self.preprocess_image(image)
                        label = steering_series[j]
                        if image_column == 'left':
                            delta_steering = -angle_offset
                        elif image_column == 'right':
                            delta_steering = angle_offset
                        else:
                            delta_steering = 0
                        
                        label = label + delta_steering
                        
                        X_train.append(image_np)
                        y_train.append(label)
                        weights.append(self.get_weight(label))
                        
                        flipped_image = np.fliplr(image_np)
                        flipped_label = -label
                        
                        X_train.append(flipped_image)
                        y_train.append(flipped_label)
                        weights.append(self.get_weight(flipped_label))
                    
                        
                    X_train, y_train, weights = shuffle(X_train, y_train, weights)
                    yield np.array(X_train), np.array(y_train), np.array(weights)


    def get_center_only_generator(self, data_folder, batch_size=64):
        driving_log_df = self.get_driving_log_dataframe(data_folder)
        driving_log_df = driving_log_df.reindex(np.random.permutation(driving_log_df.index))
        number_of_examples = len(driving_log_df)

        print(driving_log_df.head())
        
        X_train = []
        y_train = []
        weights = []
        index_in_batch = 0
        batch_number = 0
        
        angle_offset = 0.2
        
        while True:
            image_series = driving_log_df['center']
            steering_series = driving_log_df['steering']
            for offset in range(0, number_of_examples, batch_size):
                X_train = []
                y_train = []
                weights = []

                end_of_batch = min(number_of_examples, offset + batch_size)

                for j in range(offset, end_of_batch):
                    image_filename = image_series[j].lstrip().rstrip()

                    image = Image.open('{0}/{1}'.format(data_folder, image_filename))
                    image_np = self.preprocess_image(image)
                    label = steering_series[j]

                    X_train.append(image_np)
                    y_train.append(label)
                    weights.append(self.get_weight(label))

                    flipped_image = np.fliplr(image_np)
                    flipped_label = -label

                    X_train.append(flipped_image)
                    y_train.append(flipped_label)
                    weights.append(self.get_weight(flipped_label))


                X_train, y_train, weights = shuffle(X_train, y_train, weights)
                yield np.array(X_train), np.array(y_train), np.array(weights)
