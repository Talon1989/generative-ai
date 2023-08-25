import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import keras.backend as K
from Autoencoders import create_encoder
from Autoencoders import create_decoder
from Autoencoders import VAE
from transormators import covariance_matrix
import matplotlib.pyplot as plt
import os
import glob


class VanillaAutoencoder(keras.models.Model):

    def __init__(self, encoder:keras.models.Model, decoder:keras.models.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.loss_function = keras.losses.binary_crossentropy

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
        ]

    def call(self, inputs):
        z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:  # context manager
            reconstruction = self(data)
            loss = self.loss_function(data, reconstruction)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    # def train_step(self, data):
    #     x, y = data
    #     with tf.GradientTape() as tape:  # context manager
    #         reconstruction = self(x)
    #         loss = self.loss_function(reconstruction, y)
    #     grads = tape.gradient(loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     self.reconstruction_loss_tracker.update_state(loss)
    #     return {m.name: m.result() for m in self.metrics}


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


encoder, shape_before_flattening = create_encoder()
decoder = create_decoder(shape_before_flattening)
autoencoder = VanillaAutoencoder(encoder, decoder)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1/1_000))
autoencoder.fit(X_train, epochs=5, shuffle=True, batch_size=100)

print(encoder.trainable_weights[0][0][0][0][:5])  # last 5 weights of last layer
autoencoder.save('./data/z_autoencoder')
encoder.save('./data/z_encoder')
decoder.save('./data/z_decoder')
encoder = keras.models.load_model('./data/z_encoder')
print(encoder.trainable_weights[0][0][0][0][:5])  # last 5 weights of last layer































