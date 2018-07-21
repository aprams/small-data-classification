import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation


def get_top_model(n_classes=1):
    new_model = Sequential()

    # Top model structure
    #new_model.add(Dropout(0.5, input_shape=(1280,)))
    new_model.add(Dense(512, input_shape=(1280,)))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    #new_model.add(Dense(256))
    #new_model.add(BatchNormalization())
    #new_model.add(Activation('relu'))
    if n_classes == 1:
        new_model.add(Dense(1, activation='sigmoid', input_shape=(1280,)))
    else:
        new_model.add(Dense(n_classes, activation='softmax'))#, input_shape=(1280,)))

    # Fixing https://github.com/keras-team/keras/issues/2397
    graph = tf.get_default_graph()
    return new_model, graph

def get_extractor_model(image_size):
    extractor_model = keras.applications.MobileNetV2(weights="imagenet", alpha=0.5, include_top=True, input_shape=(image_size, image_size, 3))
    extractor_model.summary()
    extractor_model = keras.Model(inputs=extractor_model.input, outputs=extractor_model.get_layer('global_average_pooling2d_1').output)
    return extractor_model
