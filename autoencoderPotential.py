# making a rudimentary autoencoder which will be used for denoising and hopefully help us get higher acccuracy

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

class auto:

    def __init__(self,  nparray):
        self.nparray = nparray


    def encoder(self, ):
        encoding_dim = 32
        a = 128 *128*3
        inputshape = Input(shape=(a,))

        encoded = Dense(encoding_dim, activation="relu")(inputshape)
        decoded = Dense (a, activation="sigmoid")(encoded)

        autoencoder = Model(inputshape, decoded)

        encoder = Model(inputshape, encoded)

        encoded_input = Input(shape=(encoding_dim,))

        decoder_layer = autoencoder.layers[-1]

        decoder = Model(encoded_input, decoder_layer (encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
