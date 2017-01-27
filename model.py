from models.nvidia_pipeline import NvidiaPipeLine

pipeline = NvidiaPipeLine()

def preprocess(image):
    return pipeline.preprocess_image(image)