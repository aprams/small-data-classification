import tensorflow as tf
from keras import applications
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation

def get_model(input_shape, weights='imagenet'):
    base_model = applications.VGG16(weights=weights, include_top=False, input_shape=input_shape)

    # Got errors concatenating the models directly as it was possible in earlier versions, therefore copying layers
    new_model = Sequential()
    for l in base_model.layers:
        new_model.add(l)

    # LOCK THE TOP CONV LAYERS
    for layer in new_model.layers[:]:
        layer.trainable = False

    new_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    new_model.add(Dense(256))#, activation='relu'))
    #new_model.add(BatchNormalization(momentum=0.9))
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(1, activation='sigmoid'))

    # Fixing https://github.com/keras-team/keras/issues/2397
    graph = tf.get_default_graph()

    return new_model, graph