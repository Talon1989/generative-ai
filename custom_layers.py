import numpy as np
import tensorflow as tf
keras = tf.keras


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

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def call(self, x):
        inp_width = x.shape[3]  # channels
        if inp_width == self.width:
            residual = x
        else:
            residual = keras.layers.Conv2D(filters=self.width, kernel_size=[1, 1])(x)
        x = keras.layers.BatchNormalization(center=False, scale=False)
        x = keras.layers.Conv2D(filters=self.width, kernel_size=[3, 3],
                                padding='same', activation=keras.activations.swish)(x)
        x = keras.layers.Conv2D(filters=self.width, kernel_size=[3, 3], padding='same')(x)
        x = keras.layers.Add()([x, residual])
        return x


class DownBlock(keras.layers.Layer):

    def __init__(self, width, block_depth):
        super().__init__()
        self.width = width
        self.block_depth = block_depth

    def call(self, x):
        x, skips = x
        for _ in range(self.block_depth):
            x = ResidualBlock(width=self.width)(x)
            skips.append(x)
        x = keras.layers.AveragePooling2D(pool_size=[2, 2])(x)
        return x


class UpBlock(keras.layers.Layer):

    def __init__(self, width, block_depth):
        super().__init__()
        self.width = width
        self.block_depth = block_depth

    def call(self, x):
        x, skips = x
        x = keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(x)
        for _ in range(self.block_depth):
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width=self.width)(x)
        return x





