import tensorflow as tf
from keras import applications
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, MaxPooling2D, Convolution2D, Input
import numpy as np

def get_model(input_shape, weights='imagenet'):


    #base_model = applications.MobileNet(weights=weights, include_top=True, input_shape=input_shape, alpha=0.5)

    new_model = Sequential()

    # Got errors concatenating the models directly as it was possible in earlier versions, therefore copying layers
    #for l in base_model.layers[:-2]:#44]:
    #    new_model.add(l)

    # LOCK THE TOP CONV LAYERS
    #for layer in new_model.layers[:]:
    #    layer.trainable = False
    #new_model.add(MaxPooling2D(pool_size=(4,4)))
    #new_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    #new_model.add(Dense(256))#, activation='relu'))
    #new_model.add(BatchNormalization(momentum=0.99))
    #new_model.add(Activation('relu'))
    #new_model.add(Dropout(0.5))
    #new_model.add(Dense(64))#, activation='relu'))
    #new_model.add(BatchNormalization(momentum=0.99))
    #new_model.add(Activation('relu'))
    #new_model.add(Dropout(0.2, input_shape=(1280,)))
    #new_model.add(Dense(256, activation='elu'))
    new_model.add(Dropout(0.2, input_shape=(1280,)))
    new_model.add(Dense(1, activation='sigmoid', input_shape=(1280,)))


    # Fixing https://github.com/keras-team/keras/issues/2397
    graph = tf.get_default_graph()

    return new_model, graph