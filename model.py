from models.nvidia_pipeline import NvidiaPipeLine

from utils import get_driving_log_dataframe
from utils import get_callbacks
# Instead of implementing here, the model is in the
# models/ directory, because this allows to quickly
# switch between different pipelines
pipeline = NvidiaPipeLine()

def preprocess(image):
    return pipeline.preprocess_image(image)


def train(data_folder, validation_folder):
    model = pipeline.get_model()

    model.compile("adam", "mse")
    model.summary()

    image_generator = pipeline.get_train_generator(data_folder)
    validation_generator = pipeline.get_validation_generator(validation_folder)
    samples = pipeline.get_train_samples(get_driving_log_dataframe(data_folder))
    nb_val_samples = pipeline.get_validation_samples(get_driving_log_dataframe(validation_folder))
    callbacks_list = get_callbacks()

    model.fit_generator(image_generator, 
                        samples_per_epoch=samples, 
                        nb_epoch=20,
                       callbacks=callbacks_list,
                       validation_data=validation_generator,
                       nb_val_samples=nb_val_samples)

    model.save('model-compiled.h5')

    json_string = model.to_json()
    with open('model.json', 'w') as model_json_file:
        model_json_file.write(json_string)

    model.save_weights('model.h5')

train('data', 'valid')