"""
Adapted from https://github.com/flyyufelix/DenseNet-Keras/blob/master/densenet121.py
"""

from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from densenet_custom_layers import Scale


def create_densenet(image_size=128, number_of_channels=3, growth_rate=48, nb_filter=64, dropout_rate=0.2, classes=1, weights_path=None):
    """
    Instantiate the DenseNet-14 architecture. It only has one dense block.
    :param image_size: the dimension of the images passed as input (assumes images are square)
    :param number_of_channels: the number of channels per image
    :param growth_rate: number of filters to add per dense block
    :param nb_filter: initial number of filters
    :param dropout_rate: dropout rate
    :param classes: optional number of classes to classify images. binary classification is used if classes == 1
    :param weights_path: path to pre-trained weights
    :return: A Keras model instance
    """
    eps = 1.1e-5

    img_input = Input(shape=(image_size, image_size, number_of_channels), name='image_data')

    nb_layers = 12  # Layers for the one dense block Wide-DenseNet-14

    # Initial convolution
    X = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    X = Convolution2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(X)
    X = BatchNormalization(epsilon=eps, axis=-1, name='conv1_bn')(X)
    X = Scale(axis=-1, name='conv1_scale')(X)
    X = Activation('relu', name='relu1')(X)
    X = ZeroPadding2D((1, 1), name='pool1_zeropadding')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(X)

    X = dense_block(X, 2, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)

    X = BatchNormalization(epsilon=eps, axis=-1, name='conv2_blk_bn')(X)
    X = Scale(axis=-1, name='conv2_blk_scale')(X)
    X = Activation('relu', name='relu2_blk')(X)
    X = GlobalAveragePooling2D(name='pool2')(X)

    X = Dense(classes, name='fc3')(X)
    activation = "softmax"
    if classes == 1:
        activation = "sigmoid"
    X = Activation(activation, name='prob')(X)

    model = Model(img_input, X, name='densenet-14')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def conv_block(X, stage, branch, growth_rate, dropout_rate=None):
    """
    Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
    :param X: input tensor
    :param stage: index for dense block
    :param branch: layer index within each dense block
    :param growth_rate: growth rate of the filters
    :param dropout_rate: dropout rate
    :return: X after applying the layers listed in the description.
    """
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = growth_rate * 4
    X = BatchNormalization(epsilon=eps, axis=-1, name=conv_name_base+'_x1_bn')(X)
    X = Scale(axis=-1, name=conv_name_base+'_x1_scale')(X)
    X = Activation('relu', name=relu_name_base+'_x1')(X)
    X = Convolution2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(X)

    if dropout_rate:
        X = Dropout(dropout_rate)(X)

    # 3x3 Convolution
    X = BatchNormalization(epsilon=eps, axis=-1, name=conv_name_base+'_x2_bn')(X)
    X = Scale(axis=-1, name=conv_name_base+'_x2_scale')(X)
    X = Activation('relu', name=relu_name_base+'_x2')(X)
    X = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(X)
    X = Convolution2D(growth_rate, (3, 3), name=conv_name_base+'_x2', use_bias=False)(X)

    if dropout_rate:
        X = Dropout(dropout_rate)(X)

    return X


def transition_block(X, stage, nb_filter, dropout_rate=None):
    """
    Apply BatchNorm, 1x1 Convolution, dropout, average pooling,
    :param X: input tensor
    :param stage: index for dense block
    :param nb_filter: number of filters
    :param dropout_rate: dropout rate
    :return:
    """
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    X = BatchNormalization(epsilon=eps, axis=-1, name=conv_name_base+'_bn')(X)
    X = Scale(axis=-1, name=conv_name_base+'_scale')(X)
    X = Activation('relu', name=relu_name_base)(X)
    X = Convolution2D(nb_filter, (1, 1), name=conv_name_base, use_bias=False)(X)

    if dropout_rate:
        X = Dropout(dropout_rate)(X)

    X = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(X)

    return X


def dense_block(X, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    """
    Build a dense_block where the output of each conv_block is fed to subsequent ones
    :param X: input tensor
    :param stage: index for dense block
    :param nb_layers: the number of layers of conv_block to append to the model.
    :param nb_filter: number of filters
    :param growth_rate: number to increase nb_filter by after each layer is over
    :param dropout_rate: dropout rate
    :return: X
    """
    concat_feat = X

    for i in range(nb_layers):
        branch = i+1
        X = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate=dropout_rate)

        # Keeps a direct reference to the previous values, this is what characterises DenseNet
        concat_feat = concatenate([concat_feat, X])

        nb_filter += growth_rate

    return concat_feat
