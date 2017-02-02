import pandas as pd
import numpy as np
import cv2
import math

from PIL import Image
from sklearn.utils import shuffle


from scipy.ndimage import rotate

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
        return 1#math.exp(abs(label))

    def path_driving_log(self, data_folder):
        return '{}/driving_log.csv'.format(data_folder)


    # Credits to 
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xneaoqiwj
    def augment_brightness_camera_images(self, image):
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        image1[:,:,2] = image1[:,:,2]*random_bright
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return image1


    def random_rotation(self, image, steering_angle, rotation_amount=15):
        angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
        rad = (np.pi / 180.0) * angle
        return rotate(image, angle, reshape=False), steering_angle + (-1) * rad

    # Credits to 
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xneaoqiwj
    def add_random_shadow(self, image):
        top_y = 320*np.random.uniform()
        top_x = 0
        bot_x = 160
        bot_y = 320*np.random.uniform()
        image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        shadow_mask = 0*image_hls[:,:,1]
        X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
        shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
        #random_bright = .25+.7*np.random.uniform()
        if np.random.randint(2)==1:
            random_bright = .5
            cond1 = shadow_mask==1
            cond0 = shadow_mask==0
            if np.random.randint(2)==1:
                image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
            else:
                image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
        image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
        return image

    def crop(self, image):
        cropped_image = image[55:135, :, :]
        return cropped_image

    def resize(self, image, new_shape):
        return cv2.resize(image, new_shape)

    def get_driving_log_dataframe(self, data_folder):
        driving_log_df = pd.read_csv(self.path_driving_log(data_folder))
        return driving_log_df


    def generate_additional_image(self, image_np, label):
        toss = np.random.randint(0, 2)
        if toss == 0:
            return self.augment_brightness_camera_images(image_np), label
        elif toss == 1:
            return self.add_random_shadow(image_np), label

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
        
        angle_offset = 0.3
        
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
                        try:
                            image_filename = image_series[j].lstrip().rstrip()
                        except:
                            print(j)
                            print(image_series[j])
                            continue
                        
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

                        # generate additional image
                        X_augmented, y_augmented = self.generate_additional_image(image_np, label)
                        X_train.append(X_augmented)
                        y_train.append(y_augmented)
                        weights.append(self.get_weight(y_augmented))

                        X_augmented, y_augmented = self.generate_additional_image(flipped_image, flipped_label)
                        X_train.append(X_augmented)
                        y_train.append(y_augmented)
                        weights.append(self.get_weight(y_augmented))

                        
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
        
        angle_offset = 0.3
        
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

    def get_generator_cleaned(self, data_folder, batch_size=64):
        pass