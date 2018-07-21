import data_preprocessing
import model as mdl
import argparse
import os
import glob
import numpy as np
import keras
import json
import matplotlib.pyplot as plt
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.mobilenetv2 import preprocess_input

np.random.seed(1337)


# Training & Validation data location
TRAIN_DATA_DIR = "./data/"

# Data folders
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_SAVE_DIR = "model_save"
MODEL_FILENAME = "model.h5"
CONFIG_FILE_NAME = "config.json"

# Training constants
BATCH_SIZE = 50
LEARNING_RATE = 1e-2
IMAGE_SIZE = 224
EPOCHS = 250
AUGMENT_FACTOR = 40
TEST_SPLIT = 0.005

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

DATA_GOOD_DIR = os.path.join(args.train_dir, "good")
DATA_BAD_DIR = os.path.join(args.train_dir, "bad")

if __name__ == "__main__":
    tmp_labels = []
    tmp_images = []
    for image_path in glob.glob(os.path.join(DATA_GOOD_DIR, "*.jpeg")):
        tmp_images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (args.image_size, args.image_size))]
        tmp_labels += [1]


    for image_path in glob.glob(os.path.join(DATA_BAD_DIR, "*.jpeg")):
        tmp_images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (args.image_size, args.image_size))]
        tmp_labels += [0]

    images_train, images_val, labels_train, labels_val = train_test_split(tmp_images, tmp_labels, test_size=args.test_split, random_state=15)

    aug_images = []
    aug_labels = []

    # prefill images and labels with default data
    aug_images += images_train
    aug_labels += labels_train

    for i in range(args.augment_factor):
        aug_images += data_preprocessing.seq.augment_images(images_train)
        aug_labels.extend(labels_train)
    images_train = np.array(aug_images, dtype=np.float32)
    labels_train = aug_labels
    images_val = np.array(images_val, dtype=np.float32)

    if images_train.shape[-1] == 1 or len(images_train.shape) == 3:
        images_train = np.stack((images_train,)*3, -1)
    images_train = preprocess_input(images_train)

    if images_val.shape[-1] == 1 or len(images_val.shape) == 3:
        images_val = np.stack((images_val,)*3, -1)
    images_val = preprocess_input(images_val)

    extractor_model = mdl.get_extractor_model(args.image_size)
    bottleneck_features_train = extractor_model.predict(images_train)
    bottleneck_features_val = extractor_model.predict(images_val)
    print("Features shape:", bottleneck_features_train.shape)


    model, graph = mdl.get_top_model()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=args.learning_rate, momentum=0.9, decay=1e-6, nesterov=True), metrics=['accuracy'])

    model.fit(
        bottleneck_features_train,labels_train,
        validation_data=(bottleneck_features_val, labels_val),
        validation_steps=2,
        steps_per_epoch=len(images_train)//args.batch_size+1,
        epochs=args.epochs)

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    model.save_weights(os.path.join(args.model_save_dir, args.model_filename))
    with open(os.path.join(args.model_save_dir, CONFIG_FILE_NAME), 'w') as f:
        f.write(json.dumps(vars(args)))
