import sys
import pandas as pd
import numpy as np

from keras.callbacks import ModelCheckpoint

def img_folder(data_folder):
    return '{}/IMG'.format(data_folder)


def path_driving_log(data_folder):
    return '{}/driving_log.csv'.format(data_folder)


def get_driving_log_dataframe(data_folder):
    driving_log_df = pd.read_csv(path_driving_log(data_folder))
    return driving_log_df


def get_steering_df_in_range(df, start, end):
	return df[df['steering'] >= start][df['steering'] <= end]


def describe_data(data_folder):
	driving_log_df = get_driving_log_dataframe(data_folder)
	images_folder = img_folder(data_folder)
	class_ranges = np.linspace(-1, 1, 20)

	for first, second in zip(class_ranges, class_ranges[1:]):
		df_steering_in_range = get_steering_df_in_range(driving_log_df, first, second)
		print("Between {0} and {1}, there are {2}".format(first, second, len(df_steering_in_range)))


def get_callbacks():
	filepath = "weights-{epoch:02d}.h5"
	checkpoint = ModelCheckpoint(filepath)
	callbacks_list = [checkpoint]
	return callbacks_list

