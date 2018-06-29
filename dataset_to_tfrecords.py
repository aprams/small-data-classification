import tensorflow as tf
import os
import scipy.misc
import numpy as np
import random
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_classes_from_train_dataset(train_dir):
    return os.listdir(train_dir)


# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images

        # crop some of the images by 0-10% of their height/width
        # sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        # iaa.Affine(
        #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        #    rotate=(-10, 10),
        #    shear=(-5, 5),
        #    order=[0, 1],
        #    cval=(0, 255),
        #    mode=ia.ALL
        # ),

        # Add a value of -10 to 10 to each pixel.
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.Multiply((0.7, 1.3), per_channel=0.5),
        # Improve or worsen +the contrast of images.
        iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),
        # In some images move pixels locally around (with random
        # strengths).
        iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5
        ),
        # Change brightness of images (50-150% of original value).
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        # Same as sharpen, but for an embossing effect.
        # iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.0)),
        # Invert each image's chanell with 5% probability.
        # This sets each pixel value v to 255-v.
        # iaa.Invert(0.05, per_channel=True),  # invert color channels

        # Convert each image to grayscale and then overlay the
        # result with the original with random alpha. I.e. remove
        # colors with varying strengths.
        iaa.Grayscale(alpha=(0.0, 0.5)),

    ],
    # do all of the above augmentations in random order
    random_order=True
)


def convert_dataset(source_path, target_path, target_file, class_labels, do_augmentation=False):
    os.makedirs(target_path, exist_ok=True)
    filename = os.path.join(target_path, target_file)

    try:
        pass
        os.remove(filename)
    except OSError:
        pass

    directories = os.listdir(source_path)
    file_names = []
    labels = []
    for dir in directories:
        class_index = class_labels.index(dir)
        print(dir, "got index", class_index)
        class_image_file_names = os.listdir(os.path.join(source_path, dir))
        file_names += [os.path.join(dir, i) for i in class_image_file_names]
        labels += [class_index] * len(class_image_file_names)

    assert(len(file_names) == len(labels))

    c = list(zip(file_names, labels))

    random.shuffle(c)

    file_names, labels = zip(*c)

    # augmentor
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(file_names)):

        image = scipy.misc.imread(os.path.join(source_path, file_names[i]))
        image = scipy.misc.imresize(image,(224,224),'bilinear')
        rows = 224
        cols = 224
        depth = image.shape[2]

        if i % 100 == 0 or i == len(file_names) - 1:
            print("{0}/{1}".format(i,len(file_names)),file_names[i], labels[i])
        it_range = 10 if do_augmentation else 1
        for _ in range(it_range):
            if do_augmentation:
                augmented_image = seq.augment_image(image)
            else:
                augmented_image = image
            #plt.imshow(augmented_image)
            #plt.show()
            image_raw = augmented_image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(labels[i]),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

    out_file = open(os.path.join(training.MODEL_FINAL_SAVE_FOLDER, "model_classes.txt"), "w")
    for dir, index in zip(directories, list(range(len(directories)))):
        out_file.write("{}, {}\n".format(index, dir))
    out_file.close()
    return len(class_labels)

def convert_to_train_and_val_tfrecords(train_data_dir, val_data_dir):
    class_labels = get_classes_from_train_dataset(train_data_dir)
    if val_data_dir is not None:
        convert_dataset(val_data_dir, training.TFRECORDS_PATH, training.TFRECORDS_VAL_FILENAME, class_labels, do_augmentation=False)
    convert_dataset(train_data_dir, training.TFRECORDS_PATH, training.TFRECORDS_TRAIN_FILENAME, class_labels, do_augmentation=True)
    return len(class_labels)

if __name__ == '__main__':
    convert_to_train_and_val_tfrecords()
