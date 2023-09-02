import json
import re
import string
import numpy as np
import tensorflow as tf

keras = tf.keras
import pandas as pd
from utilities import *


# PIXEL ORDERING
  # 1  > > > > > > > >
  # 2  > > > > > > > >
  # 3  > > > > > > > >
  # 4  > > > > > > > >
  # 5  > > > > > > > >
  # 6  > > > > > > > >
  # 7  > > > > > > > >
  # 8  > > > > > > > >


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









































