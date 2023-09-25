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
        self.conv_1x1 = keras.layers.Conv2D(filters=self.width, kernel_size=[1, 1])
        self.b_n = keras.layers.BatchNormalization(center=False, scale=False)
        self.conv_1 = keras.layers.Conv2D(filters=self.width, kernel_size=[3, 3],
                                          padding='same', activation=keras.activations.swish)
        self.conv_2 = keras.layers.Conv2D(filters=self.width, kernel_size=[3, 3],
                                          padding='same', activation='linear')
        self.add = keras.layers.Add()

    def call(self, x):
        inp_width = x.shape[3]  # channels
        if inp_width == self.width:
            residual = x
        else:
            # if number of channels in input doesn't match number
            # of channels of output include and extra conv2D to bring
            # the number of  channels in line with the rest of the block
            residual = self.conv_1x1(x)
        x = self.b_n(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.add([x, residual])
        return x


class DownBlock(keras.layers.Layer):

    def __init__(self, width, block_depth):
        super().__init__()
        self.block_depth = block_depth
        self.residual_blocks = [ResidualBlock(width=width) for _ in range(block_depth)]
        self.down_sample = keras.layers.AveragePooling2D(pool_size=[2, 2])

    def call(self, x):
        x, skips_ = x
        skips = skips_.copy()
        for block in self.residual_blocks:
            x = block(x)  # increase number of channels in the image
            skips.append(x)  # and save it to a list for later use by UpBlock
        x = self.down_sample(x)  # shrink size of the image
        return [x, skips]


class UpBlock(keras.layers.Layer):

    def __init__(self, width, block_depth):
        super().__init__()
        self.block_depth = block_depth
        self.up_sample = keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concat = keras.layers.Concatenate()
        self.residual_blocks = [ResidualBlock(width=width) for _ in range(block_depth)]

    def call(self, x):
        x, skips_ = x
        skips = skips_.copy()
        x = self.up_sample(x)  # double size of the image
        for block in self.residual_blocks:
            x = self.concat([x, skips.pop()])  # get DownBlock list and concat it to current output
            x = block(x)  # reduce number of channels in the image
        return [x, skips]
