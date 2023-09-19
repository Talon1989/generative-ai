import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from call_me import SaveModel
keras = tf.keras
from utilities import *


PATH = '/home/talon/datasets/flower-dataset/dataset'


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










































































































































































