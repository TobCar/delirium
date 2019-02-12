"""
@author: Shreyansh Anand
"""
"""
making an autoencoder which will be used for denoising and help us get higher accuracy
@:param x_train_noisy - the data with noise added (10% of values are = 0) 
@:param x_train
@:param x_test_noisy - the testing data with noise added (10% of values are = 0, done with the NoiseMaker) 
@:param x_test 
"""
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def autoencoder(x_train_noisy, x_train, x_test_noisy, x_test):
    shapesize = 128*128*3
    # shape size of the images we are receiving
    inputimg = Input(shape=(shapesize,))

    net = Conv2D(32, (3,3), activation='relu', padding='same')(inputimg)
    net = MaxPooling2D((2,2), padding='same')(net)
    net = Conv2D(32, (3,3), activation='relu', padding='same')(net)
    encoded_imgs = MaxPooling2D((2,2), padding='same')(net)

    net = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_imgs)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(32, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded_imgs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)

    autoencoder = Model(inputimg, decoded_imgs)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

