import pandas as pd
from shutil import copyfile

driving_log = pd.read_csv('driving_log.csv')

def source_image_filename(filename):
	return '../{}'.format(filename)


def copy_images_from_perspective(perspective):
	for image_file in driving_log[perspective]:
		image_file = image_file.lstrip().rstrip()
		copyfile(source_image_filename(image_file), image_file)

perspectives = ['center', 'left', 'right']

for perspective in perspectives:
	copy_images_from_perspective(perspective)
