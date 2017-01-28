import sys

from models.nvidia_pipeline import NvidiaPipeLine
from models.vgg_pipeline import VGGPipeline

from utils import get_driving_log_dataframe
from utils import get_callbacks
from keras.models import load_model
# Instead of implementing here, the model is in the
# models/ directory, because this allows to quickly
# switch between different pipelines
pipeline = NvidiaPipeLine()

BATCH_SIZE = 16
EPOCHS = 5

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

    model.summary()

    image_generator = pipeline.get_train_generator(data_folder, batch_size=BATCH_SIZE)
    validation_generator = pipeline.get_validation_generator(validation_folder, batch_size=BATCH_SIZE)
    samples = pipeline.get_train_samples(get_driving_log_dataframe(data_folder))
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
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
