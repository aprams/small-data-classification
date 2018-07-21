import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob
import os
import scipy
import math
import errno
import matplotlib.pyplot as plt
import shutil


ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255)),
    iaa.Multiply((0.5, 1.5)),
    iaa.ContrastNormalization((0.5, 1.5)),
    iaa.Affine(
        #scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        #translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-45, 45),
        #shear=(-8, 8)
    )
    ], random_order=True)


def get_files_with_endings(dir, possible_endings):
    results = []
    for ending in possible_endings:
        path = os.path.join(dir, ending)
        results.extend(glob.glob(path))
    return results


def get_file_paths_from_data_folder(data_folder, possible_endings=["*.jpg", "*.jpeg", "*.png"]):
    files = []
    for dir in os.listdir(data_folder):
        full_dir_path = os.path.join(data_folder, dir)
        files.extend(get_files_with_endings(full_dir_path, possible_endings))
    files = [os.path.normpath(f) for f in files]
    return files


def generate_bottlenecks(files, bottleneck_folder, model, preprocessing_fn=None, augment_fn=None, augment_count=1, batch_size=256, image_size=224):
    for h in range(augment_count):
        # Iterate over all images in batches
        for i in range(int(math.ceil(len(files) / batch_size))):
            image_batch = []
            # Read images in batch correctly
            for j in range(batch_size):
                if i * batch_size + j >= len(files):
                    break
                file_path = files[i * batch_size + j]
                file_extension = os.path.splitext(file_path)[1][1:]
                image_batch += [
                    scipy.misc.imresize(plt.imread(file_path, format=file_extension), (image_size, image_size))]

            # resize to 3 channel, augment, preprocess
            image_batch = np.array(image_batch, dtype=np.float32)
            print("Image batch size:", image_batch.shape)
            if image_batch.shape[-1] == 1 or len(image_batch.shape) == 3:
                image_batch = np.stack((image_batch,) * 3, -1)
                print("Multi-channel image batch size:", image_batch.shape)
            #augment_fn(image_batch)
            image_batch = augment_fn(image_batch) if (augment_fn != None) and (h > 0) else image_batch
            preprocessed_images = preprocessing_fn(image_batch) if preprocessing_fn != None else image_batch
            print("preprocessed_images size:", preprocessed_images.shape)
            bottleneck_features = model.predict(preprocessed_images)

            # save bottlenecks as .npy
            for j in range(batch_size):
                if i * batch_size + j >= len(files):
                    break
                file_path = files[i * batch_size + j]
                path_components = list(file_path.split(os.sep)[-2:])
                path_components[-1] += "_" + str(h) + ".npy"
                joined_path = os.path.join(bottleneck_folder, *path_components)

                # https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
                if not os.path.exists(os.path.dirname(joined_path)):
                    try:
                        os.makedirs(os.path.dirname(joined_path))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                np.save(joined_path, bottleneck_features[j])


def generate_bottlenecks_split(data_folder, bottleneck_folder, model, preprocessing_fn=None, augment_fn=None, augment_count=1, batch_size=256, image_size=224, train_split=0.7):

    norm_bottleneck_folder = os.path.normpath(bottleneck_folder)
    train_bottleneck_folder = norm_bottleneck_folder + "_train"
    val_bottleneck_folder = norm_bottleneck_folder + "_val"
    test_bottleneck_folder = norm_bottleneck_folder + "_test"

    if os.path.exists(train_bottleneck_folder):
        shutil.rmtree(train_bottleneck_folder)
    if os.path.exists(val_bottleneck_folder):
        shutil.rmtree(val_bottleneck_folder)#
    if os.path.exists(test_bottleneck_folder):
        shutil.rmtree(test_bottleneck_folder)

    files = get_file_paths_from_data_folder(data_folder)
    n_files = len(files)
    print("Found {0} files in total".format(n_files))
    if train_split < 1.0:
        file_idx = np.random.permutation(n_files)

        train_file_idx = file_idx[:int(n_files*train_split)]
        train_files = [files[i] for i in train_file_idx]
        generate_bottlenecks(train_files, train_bottleneck_folder, model, preprocessing_fn=preprocessing_fn, augment_fn=augment_fn, augment_count=augment_count, batch_size=batch_size, image_size=image_size)

        val_file_idx = file_idx[len(train_files):int(len(train_files) + n_files * ((1.0-train_split)/2.0))]
        val_files = [files[i] for i in val_file_idx]
        generate_bottlenecks(val_files, val_bottleneck_folder, model, preprocessing_fn=preprocessing_fn, batch_size=batch_size, image_size=image_size)

        test_file_idx = file_idx[len(train_files)+len(val_files):]
        test_files = [files[i] for i in test_file_idx]
        generate_bottlenecks(test_files, test_bottleneck_folder, model, preprocessing_fn=preprocessing_fn, batch_size=batch_size, image_size=image_size)
    else:
        generate_bottlenecks(files, bottleneck_folder, model, preprocessing_fn, augment_fn, augment_count, batch_size, image_size)



if __name__ == "__main__":
    import model as mdl
    from keras.applications.mobilenetv2 import preprocess_input
    generate_bottlenecks_split("./TrainV8_in_struktur/", "./bottlenecks_train/", mdl.get_extractor_model(224), preprocess_input, augment_fn=seq.augment_images, augment_count=40)