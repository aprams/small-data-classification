import data_augment
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
from keras.applications.mobilenetv2 import preprocess_input
np.random.seed(1337)


# Training & Validation data location
TRAIN_DATA_DIR = "/home/aprams/deevio-classification/data/"

# Data folders
MODEL_TMP_SAVE_DIR = "tmp"
MODEL_FINAL_SAVE_DIR = "model"
MODEL_FILENAME = "model.h5"

# Training constants
BATCH_SIZE = 50
LEARNING_RATE = 1e-2
SAVE_INTERVAL = 1000
RECREATE_TFRECORDS = True
IMAGE_SIZE = 128
EPOCHS = 2500
AUGMENT_FACTOR = 0

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

if __name__ == "__main__":
    tmp_labels = []
    tmp_images = []
    for image_path in glob.glob(os.path.join(DATA_GOOD_DIR, "*.jpeg")):
        tmp_images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (IMAGE_SIZE, IMAGE_SIZE))]
        tmp_labels += [1]


    for image_path in glob.glob(os.path.join(DATA_BAD_DIR, "*.jpeg")):
        tmp_images += [scipy.misc.imresize(plt.imread(image_path, format='jpeg'), (IMAGE_SIZE, IMAGE_SIZE))]
        tmp_labels += [0]

    images_train, images_val, labels_train, labels_val = train_test_split(tmp_images, tmp_labels, test_size=0.1, random_state=1)

    aug_images = []
    aug_labels = []
    print(type(labels_train))
    for i in range(AUGMENT_FACTOR):
        aug_images += data_augment.seq.augment_images(images_train)
        aug_labels.extend(labels_train)
    if AUGMENT_FACTOR == 0:
        aug_images = images_train
        aug_labels = labels_train
    images_train = aug_images
    labels_train = aug_labels

    images_train = np.array(images_train, dtype=np.float32)
    images_train = np.stack((images_train,)*3, -1)
    images_train = preprocess_input(images_train)

    images_val = np.array(images_val, dtype=np.float32)
    images_val = np.stack((images_val,)*3, -1)
    images_val = preprocess_input(images_val)

    extractor_model = nail_model.get_extractor_model(IMAGE_SIZE)
    model = keras.Model(inputs=extractor_model.input, outputs=extractor_model.get_layer('global_average_pooling2d_1').output)
    bottleneck_features_train = model.predict(images_train)
    bottleneck_features_val = model.predict(images_val)
    print("Features shape:", bottleneck_features_train.shape)


    model, graph = nail_model.get_top_model()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, decay=1e-6, nesterov=True), metrics=['accuracy'])

    model.fit(
        bottleneck_features_train,labels_train,
        validation_data=(bottleneck_features_val, labels_val),
        validation_steps=2,
        steps_per_epoch=len(images_train)//BATCH_SIZE+1,
        epochs=EPOCHS)

    if not os.path.exists(MODEL_FINAL_SAVE_DIR):
        os.mkdir(MODEL_FINAL_SAVE_DIR)

    model.save_weights(os.path.join(MODEL_FINAL_SAVE_DIR, MODEL_FILENAME))

    output = model.predict(extractor_model.get_layer('global_average_pooling2d_1').output)

    from keras.preprocessing import image
    img = image.load_img("/home/aprams/deevio/deevio-classification/data/good/1522072665_good.jpeg", target_size=(model.input_shape[1], model.input_shape[2]))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    global graph
    with graph.as_default():
        result = model.predict(img)
    print(model.predict(img))

