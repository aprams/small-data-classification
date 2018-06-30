import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def get_top_model():
    new_model = Sequential()

    # Top model structure
    new_model.add(Dense(1, activation='sigmoid', input_shape=(1280,)))

    # Fixing https://github.com/keras-team/keras/issues/2397
    graph = tf.get_default_graph()
    return new_model, graph

def get_extractor_model(image_size):
    extractor_model = keras.applications.MobileNetV2(weights="imagenet", alpha=0.5, include_top=True, input_shape=(image_size, image_size, 3))
    extractor_model = keras.Model(inputs=extractor_model.input, outputs=extractor_model.get_layer('global_average_pooling2d_1').output)
    return extractor_model

#def get_model(image_size, top_model_load_path=None):
#    top_model, graph = get_top_model()
#    if top_model_load_path is not None:
#        top_model.load_weights(top_model_load_path)
#    with graph.as_default():
#        extractor_model = get_extractor_model(image_size)
#    #output = top_model(extractor_model.output)
#    model = keras.Model(inputs=extractor_model.input, outputs=top_model.output)
#    return model, graph