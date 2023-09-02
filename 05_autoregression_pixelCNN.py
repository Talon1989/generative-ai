import json
import re
import string
import numpy as np
import tensorflow as tf

keras = tf.keras
import pandas as pd
from utilities import *


# PREPPING DATA


def preprocess(imgs_int, img_size=16, pixel_levels=4):
    imgs_int = np.expand_dims(imgs_int, -1)
    imgs_int = tf.image.resize(imgs_int, (img_size, img_size)).numpy()
    imgs_int = (imgs_int / (256 / pixel_levels)).astype(int)
    imgs = imgs_int.astype("float32")
    imgs = imgs / pixel_levels
    return imgs, imgs_int


(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
input_data, output_data = preprocess(X_train)


# PIXEL ORDERING
  # 1  > > > > > > > >
  # 2  > > > > > > > >
  # 3  > > > > > > > >
  # 4  > > > > > > > >
  # 5  > > > > > > > >
  # 6  > > > > > > > >
  # 7  > > > > > > > >
  # 8  > > > > > > > >


# CUSTOM LAYERS TO BE USED


# hardcoded for greyscale images (1 channel)
class MaskedConvLayer(keras.layers.Layer):

    def __init__(self, mask_type='A', **kwargs):
        super(MaskedConvLayer, self).__init__()
        assert mask_type == 'A' or mask_type == 'B'
        self.mask_type = mask_type
        self.conv = keras.layers.Conv2D(**kwargs)
        self.mask = None

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        # kernel_shape = self.conv.kernel_size
        self.mask = np.zeros(shape=kernel_shape)
        # unmasks preceding rows of central one
        self.mask[0: kernel_shape[0] // 2, ...] = 1.
        # unmasks preceding columns of central row
        self.mask[kernel_shape[0] // 2, 0: kernel_shape[1] // 2, ...] = 1.
        if self.mask_type == 'B':
            # unmasks central pixel
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.

    def call(self, inputs):
        # masking filter weights
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        # halves # channels
        self.conv_1 = keras.layers.Conv2D(
            filters=filters//2, kernel_size=[1, 1], activation='relu'
        )
        self.pixel_conv = MaskedConvLayer(
            mask_type='B', filters=filters//2, kernel_size=[3, 3],
            activation='relu', padding='same'
        )
        # doubles # channels to match input shape
        self.conv_2 = keras.layers.Conv2D(
            filters=filters, kernel_size=[1, 1], activation='relu'
        )

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.pixel_conv(x)
        x = self.conv_2(x)
        return keras.layers.add([inputs, x])  # inputs skip connection


# PIXELCNN ARCHITECTURE


def build_pixelcnn():

    inputs = keras.layers.Input(shape=(16, 16, 1))
    x = MaskedConvLayer(mask_type='A', filters=128, kernel_size=[7, 7],
                        activation='relu', padding='same')(inputs)
    for _ in range(5):
        x = ResidualBlock(filters=2**7)(x)
    for _ in range(2):
        x = MaskedConvLayer(mask_type='B', filters=128, kernel_size=[1, 1]
                            , strides=1, activation='relu', padding='valid')(x)
    output = keras.layers.Conv2D(
        filters=4, kernel_size=[1, 1], strides=1, activation='softmax'
    )(x)
    pixel_cnn = keras.models.Model([inputs, output])
    return pixel_cnn


pixelcnn = build_pixelcnn()
pixelcnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1/2_000),
    loss=keras.losses.SparseCategoricalCrossentropy()
)


class ImageGenerator(keras.callbacks.Callback):

    def __init__(self, n_img):
        super(ImageGenerator, self).__init__()
        self.n_img = n_img

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def generate(self, temperature):
        generated_imgs = np.zeros(  # batch of empty images
            shape=(self.n_img, ) + pixelcnn.input_shape[1:]
        )
        batch, rows, cols, channels = generated_imgs.shape
        for r in range(rows):
            for c in range(cols):
                for ch in range(channels):
                    # predicts distribution of the next pixel value
                    probs = self.model.predict(generated_imgs)[:, r, c, :]
                    # sample a pixel level from the predicted distribution
                    generated_imgs[:, r, c, ch] = [
                        self.sample_from(x, temperature) for x in probs]
                    # convert pixel level to range [0, 1]
                    generated_imgs[:, r, c, ch] /= 4
        return generated_imgs

    def on_epoch_end(self, epoch, logs=None):
        generated_imgs = self.generate(temperature=1.)
        display_images(generated_imgs)


img_gen_callback = ImageGenerator(n_img=2)


# FIT THE MODEL

































