import nail_model
import argparse
import os
import glob
import numpy as np
import keras
import matplotlib.pyplot as plt
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
np.random.seed(1337)


# Training & Validation data location
TRAIN_DATA_DIR = "/home/aprams/deevio-classification/data/"

# Data folders
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_FINAL_SAVE_DIR = "model"
MODEL_FILENAME = "model.h5"

# Training constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
SAVE_INTERVAL = 1000
RECREATE_TFRECORDS = True
IMAGE_SIZE = 224
EPOCHS = 5

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch-size", default=BATCH_SIZE, type=int)
parser.add_argument("-lr", "--learning-rate", default=LEARNING_RATE, type=int)
parser.add_argument("-ep", "--epochs", default=EPOCHS, type=int,
                    help="Number of epochs until stopping the training procedure")
parser.add_argument("-rt", "--recreate-tfrecords", default=RECREATE_TFRECORDS, type=bool,
                    help="Whether to recreate tfrecord files (usually only need this once)")
parser.add_argument("-is", "--image-size", default=IMAGE_SIZE, type=int,
                    help="Preprocessed image sizes")
parser.add_argument("-si", "--save-interval", default=SAVE_INTERVAL, type=int,
                    help="Number of iterations between model saves")
parser.add_argument("-td", "--train-dir", default=TRAIN_DATA_DIR, type=str,
                    help="Training data directory containing 'good' and 'bad' class folders")
parser.add_argument("-mtd", "--model-tmp-dir", default=MODEL_TMP_SAVE_DIR, type=str,
                    help="Where to save intermediate model states to")
parser.add_argument("-mfd", "--model-final-dir", default=MODEL_FINAL_SAVE_DIR, type=str,
                    help="Where to save the final model to")
args = parser.parse_args()

print(args)

DATA_GOOD_DIR = os.path.join(TRAIN_DATA_DIR, "good")
DATA_BAD_DIR = os.path.join(TRAIN_DATA_DIR, "bad")

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

if __name__ == "__main__":

    from keras.applications.vgg16 import preprocess_input

    # KERAS HAS IMAGE.LOAD_IMG with resizing
    labels = []
    images = []
    for image_path in glob.glob(os.path.join(DATA_GOOD_DIR, "*.jpeg")):
        images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (IMAGE_SIZE, IMAGE_SIZE))]
        labels += [1]

    for image_path in glob.glob(os.path.join(DATA_BAD_DIR, "*.jpeg")):
        images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (IMAGE_SIZE, IMAGE_SIZE))]
        labels += [0]

    images = np.array(images)
    images = np.stack((images,)*3, -1)
    images = preprocess_input(images)
    labels = np.array(labels)

    print("Images shape:", images.shape)

    data_train, data_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.10, random_state=42)

    model = nail_model.get_model(input_shape)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9), metrics=['accuracy'])
    model.summary()

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True)
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(data_train, labels_train, BATCH_SIZE)
    val_generator = val_datagen.flow(data_val, labels_val, BATCH_SIZE)
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(data_train)//BATCH_SIZE,
        epochs=EPOCHS)

    if not os.path.exists(MODEL_FINAL_SAVE_DIR):
        os.mkdir(MODEL_FINAL_SAVE_DIR)
    #model.save(os.path.join(MODEL_FINAL_SAVE_DIR, MODEL_FILENAME))
    model.save_weights('model_weights.h5')
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())
    print(model.to_json())

