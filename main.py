import tensorflow as tf
import numpy as np
import os
import shutil
import glob
import argparse
import dataset_to_tfrecords

import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
from spatial_transformer import SpatialTransformer


# Training & Validation data location
TRAIN_DATA_DIR = "./TrainV3V4Merged/data/train/"
VAL_DATA_DIR = "./TrainV3V4Merged/data/validation/"#"./handselected_val/"#

# Data folders
TFRECORDS_PATH = "./tfrecords"
TFRECORDS_TRAIN_FILENAME = "train.tfrecords"
TFRECORDS_VAL_FILENAME = "validation.tfrecords"
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_FINAL_SAVE_DIR = "model"

# Training constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
SAVE_INTERVAL = 1000
ITERATIONS = 300000
RECREATE_TFRECORDS = True
IMAGE_SIZE = 256

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch-size", default=BATCH_SIZE, type=int)
parser.add_argument("-lr", "--learning-rate", default=LEARNING_RATE, type=int)
parser.add_argument("-it", "--iterations", default=ITERATIONS, type=int,
                    help="Number of iterations until stopping the training procedure")
parser.add_argument("-rt", "--recreate-tfrecords", default=RECREATE_TFRECORDS, type=bool,
                    help="Whether to recreate tfrecord files (usually only need this once)")
parser.add_argument("-is", "--image-size", default=IMAGE_SIZE, type=int,
                    help="Preprocessed image sizes")
parser.add_argument("-si", "--save-interval", default=SAVE_INTERVAL, type=int,
                    help="Number of iterations between model saves")
parser.add_argument("-td", "--train-dir", default=TRAIN_DATA_DIR, type=str,
                    help="Training data directory containing class folders")
parser.add_argument("-vd", "--val-dir", default=VAL_DATA_DIR, type=str,
                    help="Validation data directory containing class folders")
parser.add_argument("-mtd", "--model-tmp-dir", default=MODEL_TMP_SAVE_DIR, type=str,
                    help="Where to save intermediate model states to")
parser.add_argument("-mfd", "--model-final-dir", default=MODEL_FINAL_SAVE_DIR, type=str,
                    help="Where to save the final model to")

args = parser.parse_args()

print(args)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], out_type=tf.uint8)
    height = tf.cast(features['height'], dtype=tf.int32)
    width = tf.cast(features['width'], dtype=tf.int32)
    depth = tf.cast(features['depth'], dtype=tf.int32)
    label = tf.cast(features['label'], dtype=tf.int32)
    height = tf.Print(height, [height], message="height")
    print(height, depth, width)
    # cast image int64 to float32 [0, 255] -> [-0.5, 0.5]
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image_shape = tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.reshape(image, image_shape)

    return image, label

def inputs(filename, batch_size):
    filenames = [filename]
    print(filenames)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames)
    print(filename_queue)
    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=100)

    return images, labels

def preprocess(image):
    return dataset_to_tfrecords.seq.augment_image(image)

if __name__ == "__main__":
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))

    model = Sequential()

    model.add(SpatialTransformer(localization_net=locnet,
                                 output_size=(30, 30), input_shape=input_shape))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
        #preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        'data/',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=32,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)



