import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from call_me import SaveModel
keras = tf.keras
from utilities import *
from custom_layers import (ResidualBlock, UpBlock, DownBlock)


PATH = '/home/talon/datasets/flower-dataset/dataset'
EMA = 999/1_000


train_data = keras.utils.image_dataset_from_directory(
    directory=PATH, labels=None, image_size=[64, 64],
    batch_size=None, shuffle=True, seed=42, interpolation='bilinear'
)


# scale to range [0, 1]
def preprocess(img):
    return tf.cast(img, 'float32') / 255.


train = train_data.map(lambda x: preprocess(x))
train = train.repeat(5)  # repeat dataset 5 times
train = train.batch(2**6, drop_remainder=True)


def linear_diffusion_schedule(diffusion_times):
    min_rate = 1/10_000
    max_rate = 1/50
    betas = min_rate + tf.convert_to_tensor(diffusion_times) * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)  # cumulative product
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 1/50
    max_signate_rate = 19/20
    start_angle = tf.acos(max_signate_rate)
    end_angle = tf.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)
    return noise_rates, signal_rates


T = 1_000
diff_times = [x/T for x in range(T)]
linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(diff_times)


class DiffusionModel(keras.models.Model):

    def __init__(self, model: keras.models.Model, diff_schedule):
        super().__init__()
        self.normalizer = keras.layers.Normalization()
        self.network = model
        self.ema_network = keras.models.clone_model(self.network)
        self.diffusion_schedule = diff_schedule

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates ** 2], training=training
        )
        pred_images = (noisy_images - (noise_rates * pred_noises)) / signal_rates
        return pred_noises, pred_images

    def train_step(self, data):

        images = data

        # normalize batch of images : they have zero mean and unit variance
        images = self.normalizer(images, training=True)
        # sample noise to match the shape of the input images
        noises = tf.random.normal(shape=tf.shape(images))
        batch_size = tf.shape(images)[0]

        diffusion_times = tf.random.uniform(
            shape=[batch_size, 1, 1, 1], minval=0., maxval=1.
        )  # sample random diffusion times
        # and use them to generate noise and signal rates
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = (signal_rates * images) + (noise_rates * noises)

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )
            noise_loss = self.loss(noises, pred_noises)  # MSE
        grads = tape.gradient(noise_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        self.noise_loss_tracker.update(noise_loss)

        # EMA (soft) update
        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.resut() for m in self.metrics}







































































































































































