import data_preprocessing
import model as mdl
import argparse
import os
import glob
import numpy as np
import keras
import json
import math
import random
import matplotlib.pyplot as plt
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.mobilenetv2 import preprocess_input
import data_preprocessing
from keras.callbacks import EarlyStopping
import shutil
np.random.seed(1337)


# Training & Validation data location
REGENERATE_DATA = False
TRAIN_DATA_DIR = "./TrainV8_in_struktur/"
VAL_DATA_DIR = "./ValV8_in_struktur/"
TEST_DATA_DIR = "./TestV8_in_struktur/"
BOTTLENECK_DATA_DIR = "./bottlenecks"
TRAIN_SUFFIX = "_train"
VAL_SUFFIX = "_val"
TEST_SUFFIX = "_test"

# Data folders
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_SAVE_DIR = "model_save"
MODEL_FILENAME = "model.h5"
CONFIG_FILE_NAME = "config.json"

# Training constants
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
EPOCHS = 3000
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

    train_bottleneck_dir = BOTTLENECK_DATA_DIR + TRAIN_SUFFIX
    val_bottleneck_dir = BOTTLENECK_DATA_DIR + VAL_SUFFIX
    test_bottleneck_dir = BOTTLENECK_DATA_DIR + TEST_SUFFIX


    if REGENERATE_DATA:
        if os.path.exists(train_bottleneck_dir):
            shutil.rmtree(train_bottleneck_dir)
        if os.path.exists(val_bottleneck_dir):
            shutil.rmtree(val_bottleneck_dir)  #
        if os.path.exists(test_bottleneck_dir):
            shutil.rmtree(test_bottleneck_dir)

        extractor_model = mdl.get_extractor_model(224)

        train_files = data_preprocessing.get_file_paths_from_data_folder(TRAIN_DATA_DIR)
        data_preprocessing.generate_bottlenecks(train_files, train_bottleneck_dir,
                                                extractor_model,
                                                preprocess_input,
                                                augment_fn=data_preprocessing.seq.augment_images, augment_count=20)
        if VAL_DATA_DIR != "":
            val_files = data_preprocessing.get_file_paths_from_data_folder(VAL_DATA_DIR)
            data_preprocessing.generate_bottlenecks(val_files, val_bottleneck_dir,
                                                    extractor_model,
                                                    preprocess_input)
        if TEST_DATA_DIR != "":
            test_files = data_preprocessing.get_file_paths_from_data_folder(TEST_DATA_DIR)
            data_preprocessing.generate_bottlenecks(test_files, test_bottleneck_dir,
                                                    extractor_model,
                                                    preprocess_input)
    train_dirs = os.listdir(train_bottleneck_dir)

    use_validation = False
    if os.path.exists(val_bottleneck_dir):
        val_dirs = os.listdir(val_bottleneck_dir)
        use_validation = True

    use_testing = False
    if os.path.exists(test_bottleneck_dir):
        test_dirs = os.listdir(test_bottleneck_dir)
        use_testing = True

    n_classes = len(train_dirs)
    with open(os.path.join(MODEL_SAVE_DIR, 'classes.txt'), 'w') as f:
        for dir in train_dirs:
            f.write(dir)

    model, graph = mdl.get_top_model(n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=args.learning_rate), metrics=['accuracy'])#, momentum=0.9, decay=1e-6, nesterov=True), metrics=['accuracy'])

    def generator(path, batch_size):
        # TODO: Make this a parallelized operation
        file_paths = data_preprocessing.get_file_paths_from_data_folder(path, possible_endings=["*.npy"])
        unique_dir_names = list(set([os.path.dirname(file_paths[i]) for i in range(len(file_paths))]))
        random.shuffle(unique_dir_names)
        #print(file_paths)
        cur_idx = 0
        while(True):
            batch_data = []
            batch_labels = []
            for i in range(batch_size):
                batch_data += [np.load(file_paths[(cur_idx+i) % len(file_paths)])]
                batch_labels += [unique_dir_names.index(os.path.dirname(file_paths[(cur_idx+i)%len(file_paths)]))]
            cur_idx += batch_size
            if cur_idx > len(file_paths):
                cur_idx = cur_idx % len(file_paths)
                random.shuffle(unique_dir_names)
            yield (np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels), num_classes=n_classes))

    train_data_gen = generator(train_bottleneck_dir, batch_size=BATCH_SIZE)

    early_stopping_callback = EarlyStopping(patience=10)
    fit_args = {'generator': train_data_gen,
                'epochs' : args.epochs,
                'steps_per_epoch': int(math.ceil(len(data_preprocessing.get_file_paths_from_data_folder(
                    train_bottleneck_dir, possible_endings=["*.npy"]))/BATCH_SIZE))}
                #'callbacks':[early_stopping_callback]}

    if use_validation:
        val_data_gen = generator(val_bottleneck_dir, batch_size=BATCH_SIZE)
        fit_args['validation_data'] = val_data_gen
        fit_args['validation_steps'] = int(math.ceil(len(data_preprocessing.get_file_paths_from_data_folder(
                    val_bottleneck_dir, possible_endings=["*.npy"]))/BATCH_SIZE))

    model.fit_generator(
        **fit_args
    )


    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    model.save_weights(os.path.join(args.model_save_dir, args.model_filename))
    with open(os.path.join(args.model_save_dir, CONFIG_FILE_NAME), 'w') as f:
        f.write(json.dumps(vars(args)))
