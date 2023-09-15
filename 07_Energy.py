import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from call_me import SaveModel

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


def generate_samples(model: EnergyBasedModel, inp_imgs, n_steps, step_size, noise=1 / 200):
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


class Buffer:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.examples = [
            tf.random.uniform(shape=[1, 32, 32, 1]) * 2 - 1
            for _ in range(128)
        ]  # batch of random noise

    def sample_new_examples(self, n_steps, step_size, noise=1 / 200):

        def generate_samples():
            nonlocal inp_images, n_steps, step_size, noise
            for _ in range(n_steps):
                # noise em up
                inp_images += tf.random.normal(inp_images.shape, mean=0, stddev=noise)
                inp_images = tf.clip_by_value(inp_images, clip_value_min=-1., clip_value_max=+1.)
                with tf.GradientTape() as tape:
                    tape.watch(inp_images)
                    score = self.model(inp_images)
                grads = - tape.gradient(score, inp_images)
                grads = tf.clip_by_value(grads, -0.03, +0.03)  # let's not move too much
                inp_images += - step_size * grads
                inp_images = tf.clip_by_value(inp_images, -1., +1.)
                return inp_images

        # 5% of observations will be generated from scratch
        n_new = np.random.binomial(n=128, p=1 / 20)
        # generation of observations
        rand_images = tf.random.uniform(shape=[n_new, 32, 32, 1]) * 2 - 1
        old_images = tf.concat(
            random.choices(self.examples, k=128 - n_new), axis=0
        )  # concatenation of some random buffered images and generated ones
        inp_images = tf.concat([rand_images, old_images], axis=0)
        # observations are run through Langevin sampler
        inp_images = generate_samples()
        # then they are added to the buffer
        self.examples = tf.split(inp_images, 128, axis=0) + self.examples
        # and trimmed to a max length of 8192 observations
        self.examples = self.examples[:8192]
        return inp_images


class EBM(keras.models.Model):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.buffer = Buffer(self.model)
        self.alpha = 1/10
        self.loss_metric = keras.metrics.Mean(name='loss')
        self.reg_loss_metric = keras.metrics.Mean(name='reg')
        self.cdiv_loss_metric = keras.metrics.Mean(name='cdiv')
        self.real_out_metric = keras.metrics.Mean(name='real')
        self.fake_out_metric = keras.metrics.Mean(name='fake')

    @property
    def metrics(self):
        return[
            self.loss_metric,
            self.reg_loss_metric,
            self.cdiv_loss_metric,
            self.real_out_metric,
            self.fake_out_metric
        ]

    def train_step(self, data):  # data are real images
        data += tf.random.normal(
            shape=tf.shape(data), mean=0, stddev=1/200
        )
        real_images = tf.clip_by_value(data, -1., 1.)
        fake_images = self.buffer.sample_new_examples(
            n_steps=60, step_size=10
        )
        inp_images = tf.concat([real_images, fake_images], axis=0)
        with tf.GradientTape() as tape:
            # real and fake scores
            real_out, fake_out = tf.split(self.model(inp_images), 2, axis=0)
            # contrastive divergence loss
            cdiv_loss = tf.reduce_mean(fake_out, axis=0) - tf.reduce_mean(real_out, axis=0)
            # regularization loss
            reg_loss = self.alpha * tf.reduce_mean(real_out ** 2 + fake_out ** 2, axis=0)
            loss = cdiv_loss + reg_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        self.reg_loss_metric.update_state(reg_loss)
        self.cdiv_loss_metric.update_state(cdiv_loss)
        self.real_out_metric(tf.reduce_mean(real_out, axis=0))
        self.fake_out_metric(tf.reduce_mean(fake_out, axis=0))

    # for validation
    def test_step(self, data):  # data are real images
        real_images = data
        batch_size = real_images.shape[0]
        # tf.random.uniform distribution is default to be U([0, 1))
        fake_images = tf.random.uniform([batch_size, 32, 32, 1]) * 2 - 1
        inp_images = tf.concat([real_images, fake_images], axis=0)
        real_out, fake_out = tf.split(self.model(inp_images), 2, axis=0)
        cdiv_loss = tf.reduce_mean(fake_out, axis=0) - tf.reduce_mean(real_out, axis=0)
        self.cdiv_loss_metric.update_state(cdiv_loss)
        self.real_out_metric(tf.reduce_mean(real_out, axis=0))
        self.fake_out_metric(tf.reduce_mean(fake_out, axis=0))
        return {m.name: m.result() for m in self.metrics[2:]}


ebm = EBM(model_)
ebm.compile(optimizer=keras.optimizers.Adam(learning_rate=1/10_000), run_eagerly=True)
save_model_callback = SaveModel(ebm, 'data/models.EBM')
ebm.fit(X_train_dataset, epochs=60, validation_data=X_test_dataset, callbacks=[save_model_callback])
ebm.save('data/models/EBM_00')
#




















































































































