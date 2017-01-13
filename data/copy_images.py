import sys
import os
import pandas as pd

from shutil import copyfile


# IMG_FOLDER is a folder that contains IMG folder, not th
# folder itself

def source_image_filename(source_img_folder, filename):
	return '{0}/{1}'.format(source_img_folder, filename)


def copy_images_from_perspective(source_img_folder, destination_img_folder, driving_log, perspective):
	for image_file in driving_log[perspective]:
		image_file = image_file.lstrip().rstrip()
		output_image_file = '{0}/{1}'.format(destination_img_folder, image_file)
		copyfile(source_image_filename(source_img_folder, image_file), output_image_file)


def copy_images(SRC_IMG_FOLDER, DEST_IMG_FOLDER, PATH_TO_DRIVING_LOG):
	driving_log = pd.read_csv(PATH_TO_DRIVING_LOG)

	perspectives = ['center', 'left', 'right']

	if not os.path.exists('{}/IMG'.format(DEST_IMG_FOLDER)):
	    os.makedirs('{}/IMG'.format(DEST_IMG_FOLDER))

	for perspective in perspectives:
		copy_images_from_perspective(SRC_IMG_FOLDER, DEST_IMG_FOLDER, driving_log, perspective)

if __name__ == "__main__":
	IMG_FOLDER = '..' if not len(sys.argv) >= 2 else sys.argv[1]
	DEST_IMG_FOLDER = '.' if not len(sys.argv) >= 3 else sys.argv[2]
	PATH_TO_DRIVING_LOG = 'driving_log.csv' if not len(sys.argv) >= 4 else sys.argv[3]
	copy_images(IMG_FOLDER, DEST_IMG_FOLDER, PATH_TO_DRIVING_LOG)