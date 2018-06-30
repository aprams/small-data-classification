import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5, deterministic=True), # horizontal flips
    iaa.Flipud(0.5),
    #iaa.Sometimes(0.5,
    #    iaa.GaussianBlur(sigma=(0, 0.5))
    #),
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255)),
    iaa.Multiply((0.8, 1.2)),
    #iaa.Affine(
    #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    #    rotate=(-180, 180),
    #    shear=(-8, 8)
    #)
    ], random_order=True)
