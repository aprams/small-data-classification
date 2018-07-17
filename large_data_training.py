import data_preprocessing
import model as mdl
import argparse
import os
import glob
import numpy as np
import keras
import json
import math
import matplotlib.pyplot as plt
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.mobilenetv2 import preprocess_input
np.random.seed(1337)


# Training & Validation data location
TRAIN_DATA_DIR = "./data/"
BOTTLENECK_DATA_DIR = "./bottlenecks"

# Data folders
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_SAVE_DIR = "model_save"
MODEL_FILENAME = "model.h5"
CONFIG_FILE_NAME = "config.json"

# Training constants
BATCH_SIZE = 50
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
EPOCHS = 250
AUGMENT_FACTOR = 40
TEST_SPLIT = 0.2

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch-size", default=BATCH_SIZE, type=int)
parser.add_argument("-lr", "--learning-rate", default=LEARNING_RATE, type=float)
parser.add_argument("-ep", "--epochs", default=EPOCHS, type=int,
                    help="Number of epochs until stopping the training procedure")
parser.add_argument("-is", "--image-size", default=IMAGE_SIZE, type=int,
                    help="Preprocessed image sizes")
parser.add_argument("-td", "--train-dir", default=TRAIN_DATA_DIR, type=str,
                    help="Training data directory containing 'good' and 'bad' class folders")
parser.add_argument("-mfd", "--model-save-dir", default=MODEL_SAVE_DIR, type=str,
                    help="Where to save the final model to")
parser.add_argument("-mfn", "--model-filename", default=MODEL_FILENAME, type=str,
                    help="final model filename (should end with .h5)")
parser.add_argument("-aug", "--augment-factor", default=AUGMENT_FACTOR, type=int,
                    help="Iterations of Data augmentations, 0 for no augmentation")
parser.add_argument("-ts", "--test-split", default=TEST_SPLIT, type=int,
                    help="Split of the data that is used for validation")
args = parser.parse_args()

print(args)


if __name__ == "__main__":

    train_dirs = os.listdir(BOTTLENECK_DATA_DIR + "_train")
    n_classes = len(train_dirs)

    model, graph = mdl.get_top_model(n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=args.learning_rate, momentum=0.9, decay=1e-6, nesterov=True), metrics=['accuracy'])



    def generator(path, batch_size):

        file_paths = data_preprocessing.get_file_paths_from_data_folder(path, possible_endings=["*.npy"])
        unique_dir_names = list(set([os.path.dirname(file_paths[i]) for i in range(len(file_paths))]))
        print(file_paths)
        cur_idx = 0
        while(True):
            batch_data = []
            batch_labels = []
            for i in range(batch_size):
                batch_data += [np.load(file_paths[(cur_idx+i) % len(file_paths)])]
                batch_labels += [unique_dir_names.index(os.path.dirname(file_paths[(cur_idx+i)%len(file_paths)]))]
            cur_idx += batch_size
            cur_idx = cur_idx % len(file_paths)
            yield (np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels), num_classes=n_classes))

    data_gen = generator("bottlenecks_train", batch_size=BATCH_SIZE)
    model.fit_generator(
        data_gen,
        epochs=args.epochs,
    steps_per_epoch=int(math.ceil(len(data_preprocessing.get_file_paths_from_data_folder("bottlenecks_train", possible_endings=["*.npy"]))/BATCH_SIZE)))

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    model.save_weights(os.path.join(args.model_save_dir, args.model_filename))
    with open(os.path.join(args.model_save_dir, CONFIG_FILE_NAME), 'w') as f:
        f.write(json.dumps(vars(args)))
