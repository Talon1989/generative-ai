import numpy as np
import tensorflow as tf
keras = tf.keras


class Discriminator(keras.models.Model):

    def __init__(self, image_size=(64, 64, 1)):
        super().__init__()
        self.layer_1 = keras.layers.Conv2D(
            filters=64, kernel_size=[4, 4], strides=2, padding='same', use_bias=False, input_shape=image_size
        )
        self.layer_2 = keras.layers.Conv2D(128, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_3 = keras.layers.Conv2D(256, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_4 = keras.layers.Conv2D(512, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_5 = keras.layers.Conv2D(1, kernel_size=[4, 4], strides=1, padding='valid', use_bias=False, activation='sigmoid')
        self.batch_norm_1 = keras.layers.BatchNormalization(momentum=9/10)
        self.batch_norm_2 = keras.layers.BatchNormalization(momentum=9/10)
        self.batch_norm_3 = keras.layers.BatchNormalization(momentum=9/10)
        self.leaky_relu = keras.layers.LeakyReLU(alpha=1/5)
        self.dropout = keras.layers.Dropout(rate=3/10)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.batch_norm_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        x = self.batch_norm_2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer_4(x)
        x = self.batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer_5(x)
        x = self.flatten(x)
        return x


class Generator(keras.models.Model):

    def __init__(self, image_size=(64, 64, 1)):
        super().__init__()
        # input shape is a 100 element vector sampled from multivariate standard normal distribution
        self.reshape = keras.layers.Reshape(target_shape=[1, 1, 100], input_shape=(100, ))
        self.layer_1 = keras.layers.Conv2DTranspose(filters=512, kernel_size=[4, 4], strides=1, padding='valid', use_bias=False)
        self.layer_2 = keras.layers.Conv2DTranspose(256, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_3 = keras.layers.Conv2DTranspose(128, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_4 = keras.layers.Conv2DTranspose(64, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_5 = keras.layers.Conv2DTranspose(1, kernel_size=[4, 4], strides=2, padding='same', use_bias=False, activation='tanh')
        self.batch_norm_1 = keras.layers.BatchNormalization(momentum=9/10)
        self.batch_norm_2 = keras.layers.BatchNormalization(momentum=9/10)
        self.batch_norm_3 = keras.layers.BatchNormalization(momentum=9/10)
        self.batch_norm_4 = keras.layers.BatchNormalization(momentum=9/10)
        self.leaky_relu = keras.layers.LeakyReLU(alpha=1/5)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.layer_1(x)
        x = self.batch_norm_1(x)
        x = self.leaky_relu(x)
        x = self.layer_2(x)
        x = self.batch_norm_2(x)
        x = self.leaky_relu(x)
        x = self.layer_3(x)
        x = self.batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.layer_4(x)
        x = self.batch_norm_4(x)
        x = self.leaky_relu(x)
        x = self.layer_5(x)
        return x



