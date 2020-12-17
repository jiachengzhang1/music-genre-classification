import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers import GaussianNoise


def _conv_block(num_filters, kernel_size=(3, 3), activation="relu", pool_size=(2, 2)):
    def f(x):
        # x = GaussianNoise(0.01)(x)
        x = Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same')(x)
        x = Activation(activation)(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_size)(x)
        x = Dropout(0.25)(x)
        return x
    return f


def _identity_conv_block(num_filters, kernel_size=(3, 3), activation="relu", pool_size=(2, 2), noise=None, batch_normal=True):
    bn_axis = 1
    if K.image_data_format() == "channels_last":
        bn_axis = 3

    def f(input):
        x = Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same')(input)
        x = Activation(activation)(x)
        if batch_normal:
            x = BatchNormalization(axis=bn_axis)(x)


        if noise:
            x = GaussianNoise(noise)(x)


        x = Conv2D(num_filters, kernel_size, strides=(1, 1),  padding='same')(x)
        x = Activation(activation)(x)
        if batch_normal:
            x = BatchNormalization(axis=bn_axis)(x)

        
        shortcut = Conv2D(filters=K.int_shape(x)[bn_axis], kernel_size=(1, 1))(input)

        x = add([x, shortcut])
        
        return x
    return f


def _conv_layers(conv_configs):
    def f(x):
        for config in conv_configs:
            if config["resNet"]:
                x = _identity_conv_block(config["num_filters"],
                                         config["kernel_size"],
                                         config["activation"],
                                         config["pool_size"],
                                         noise=config["noise"],
                                         batch_normal=config["batch_normal"])(x)
            else:
                x = _conv_block(config["num_filters"],
                                config["kernel_size"],
                                config["activation"],
                                config["pool_size"])(x)
        return x
    return f


def _classifier(num_labels, config):
    def f(x):
        if config["basic"]:
            x = Flatten()(x)
            x = Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
        else:
            x = AveragePooling2D()(x)
            x = Flatten()(x)

        x = Dense(num_labels, activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)

        return x
    return f


def MusicRecNet(input_shape, conv_configs, num_labels=10):
    inputs = Input(shape=input_shape)
    x = _conv_layers(conv_configs)(inputs)
    predictions = _classifier(num_labels, conv_configs[0])(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model