import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import pandas as pd
from utilities import *


def preprocess(imgs):
    imgs = (imgs.astype('float32') - 127.5) / 127.5
    # adds creates new elements at edges and give them -1. value
    imgs = np.pad(imgs, [[0, 0], [2, 2], [2, 2]],
                  mode='constant', constant_values=-1.)
    imgs = np.expand_dims(imgs, -1)
    return imgs


(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = preprocess(X_train)
X_test = preprocess(X_test)
X_train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
X_test_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(128)


def build_energy_function_network():
    ebm_input = keras.layers.Input(shape=(32, 32, 1))
    x = keras.layers.Conv2D(16, kernel_size=[5, 5], strides=2,
                            padding='same', activation='swish')(ebm_input)
    x = keras.layers.Conv2D(32, kernel_size=[3, 3], strides=2,
                            padding='same', activation='swish')(x)
    x = keras.layers.Conv2D(64, kernel_size=[3, 3], strides=2,
                            padding='same', activation='swish')(x)
    x = keras.layers.Conv2D(64, kernel_size=[3, 3], strides=2,
                            padding='same', activation='swish')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=64, activation='swish')(x)
    ebm_output = keras.layers.Dense(units=1, activation='linear')(x)
    return keras.models.Model(ebm_input, ebm_output)


class EnergyBasedModel(keras.models.Model):

    def __init__(self):
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(
            filters=16, kernel_size=[5, 5], strides=2, padding='same',
            activation='swish', input_shape=(None, 32, 32, 1)
        )
        self.conv_2 = keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=2, padding='same',
            activation='swish')
        self.conv_3 = keras.layers.Conv2D(
            filters=64, kernel_size=[3, 3], strides=2, padding='same',
            activation='swish')
        self.conv_4 = keras.layers.Conv2D(
            filters=64, kernel_size=[3, 3], strides=2, padding='same',
            activation='swish')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.conv_4(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        return x


# model_2 = build_energy_function_network()


model_ = EnergyBasedModel()
model_.build(input_shape=(None, 32, 32, 1))


def generate_samples(model: EnergyBasedModel, inp_imgs, n_steps, step_size, noise=1/200):
    imgs_per_step = []
    for _ in range(n_steps):
        # noise em up
        inp_imgs += tf.random.normal(inp_imgs.shape, mean=0, stddev=noise)
        inp_imgs = tf.clip_by_value(inp_imgs, clip_value_min=-1., clip_value_max=+1.)
        with tf.GradientTape() as tape:
            tape.watch(inp_imgs)
            score = model(inp_imgs)
        grads = - tape.gradient(score, inp_imgs)
        grads = tf.clip_by_value(grads, -0.03, +0.03)  # let's not move too much
        inp_imgs += -step_size * grads
        inp_imgs = tf.clip_by_value(inp_imgs, -1., +1.)
        return inp_imgs






















































































