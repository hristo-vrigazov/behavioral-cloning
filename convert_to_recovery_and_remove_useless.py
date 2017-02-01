import sys
import os
import pandas as pd

from utils import get_driving_log_dataframe
from utils import img_folder

from shutil import copyfile

directory = sys.argv[1]
output_directory = sys.argv[2]

driving_log_df = get_driving_log_dataframe(directory)


steering = driving_log_df['steering']

angle_offset = 0.3

# if image_column == 'left':
#     delta_steering = -angle_offset
# elif image_column == 'right':
#     delta_steering = angle_offset
# else:
#     delta_steering = 0

def save_images(offset_sign, image_series, angles_file):
	for i in range(len(driving_log_df)):
		if pd.isnull(image_series[i]):
			continue
		delta_steering = offset_sign * angle_offset
		image_name = image_series[i].lstrip().rstrip()
		steering_angle = steering[i] + delta_steering
		print('{0} -> {1}'.format(image_name, steering_angle))
		src_path_to_image = '{0}/{1}'.format(directory, image_name)
		dest_path_to_image = '{0}/{1}'.format(output_directory, image_name)
		copyfile(src_path_to_image, dest_path_to_image)
		angles_file.write('{0},{1}\n'.format(image_name, steering_angle))

def copy_if_has_column(column_name, steering_sign, angles_file):
	print(driving_log_df.columns)
	if column_name in driving_log_df.columns:
		image_series = driving_log_df[column_name]
		save_images(steering_sign, image_series, angles_file)

if not os.path.exists(output_directory):
	os.makedirs(output_directory)

if not os.path.exists(img_folder(output_directory)):
	os.makedirs(img_folder(output_directory))

with open('{0}/{1}'.format(output_directory, 'angles.csv'), 'a+') as angles_file:
	copy_if_has_column('left', -1, angles_file)
	copy_if_has_column('right', 1, angles_file)
	copy_if_has_column('center', 0, angles_file)