from keras import applications
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout


def get_model(input_shape):

    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Got errors concatenating the models directly as it was possible in earlier versions, therefore copying layers
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)

    new_model.add(Flatten(input_shape=model.output_shape[1:]))
    new_model.add(Dense(256, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(1, activation='sigmoid'))

    # LOCK THE TOP CONV LAYERS
    for layer in model.layers:
        layer.trainable = False

    model = new_model
    return model