import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import keras.backend as K
from Autoencoders import Encoder, Decoder
from Autoencoders import create_autoencoder
from Autoencoders import create_variational_encoder
from Autoencoders import create_decoder
from Autoencoders import VAE
from transormators import covariance_matrix
import matplotlib.pyplot as plt
import os
import glob


(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
train_indices = np.random.choice(X_train.shape[0], size=2_000, replace=False)
test_indices = np.random.choice(X_test.shape[0], size=1_000, replace=False)
X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]


def preprocess(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs, [[0, 0], [2, 2], [2, 2]], constant_values=0.)
    imgs = np.expand_dims(imgs, -1)
    return imgs


X_train = preprocess(X_train)
X_test = preprocess(X_test)

encoder = Encoder()
latent_space, output_shape = encoder(X_train)

decoder = Decoder(output_shape)
