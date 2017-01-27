import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from copy_images import copy_images

PATH_TO_DATA = '.' if not len(sys.argv) == 2 else sys.argv[1]

IMG_FOLDER = '{}/IMG'.format(PATH_TO_DATA)
PATH_TO_DRIVING_LOG = '{}/driving_log.csv'.format(PATH_TO_DATA)

driving_log_df = pd.read_csv(PATH_TO_DRIVING_LOG)

train_df, test_df =  train_test_split(driving_log_df, test_size = 0.1)

train_df, validation_df = train_test_split(train_df, test_size = 0.1)

print(len(train_df))
print(len(test_df))
print(len(validation_df))

dataframes = {'train': train_df, 'test': test_df, 'valid': validation_df}

folders = {}
folders['train'] = '{}/train'.format(PATH_TO_DATA)
folders['test'] = '{}/test'.format(PATH_TO_DATA)
folders['valid'] = '{}/valid'.format(PATH_TO_DATA)

def create_folder_if_does_not_exist(folder):
	if not os.path.exists(folder):
	    os.makedirs(folder)

for dataset in folders:
	create_folder_if_does_not_exist(dataset)
	path_to_driving_log_dataset = '{}/driving_log.csv'.format(folders[dataset])
	dataframes[dataset].to_csv(path_to_driving_log_dataset, index=False)
	copy_images(PATH_TO_DATA, folders[dataset], path_to_driving_log_dataset)