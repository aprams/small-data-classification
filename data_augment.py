import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

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
    #iaa.Affine(
    #    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    #    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    #    rotate=(-5, 5)
    #    shear=(-8, 8)
    #)
    ], random_order=True)
