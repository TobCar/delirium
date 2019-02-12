"""
@author: Tobias Carryer
"""

from keras import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dense


def create_model(compound_image_size, number_of_channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(compound_image_size, compound_image_size, number_of_channels)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), input_shape=(compound_image_size, compound_image_size, number_of_channels)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model
