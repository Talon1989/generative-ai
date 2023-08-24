import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras
import keras.backend as K


class Encoder(keras.Model):

    def __init__(self):
        super().__init__()
        # self.encoder_input = keras.layers.Input(shape=(32, 32, 1), name='encoder_input')
        self.layer_1 = keras.layers.Conv2D(32, [3, 3], strides=2, activation='relu', padding='same', input_shape=(32, 32, 1))
        self.layer_2 = keras.layers.Conv2D(64, [3, 3], strides=2, activation='relu', padding='same')
        self.layer_3 = keras.layers.Conv2D(128, [3, 3], strides=2, activation='relu', padding='same')
        self.flatten = keras.layers.Flatten()
        self.encoder_output = keras.layers.Dense(units=2, name='encoder_output')

    def call(self, inputs, training=None, mask=None):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        shape_before_flattening = K.int_shape(x)[1:]
        x = self.flatten(x)
        return self.encoder_output(x), shape_before_flattening


class Decoder(keras.Model):

    def __init__(self, shape_before_flattening):
        super().__init__()
        self.dense = keras.layers.Dense(np.prod(shape_before_flattening), input_shape=(2,))
        self.reshape = keras.layers.Reshape(target_shape=shape_before_flattening)
        self.layer_1 = keras.layers.Conv2DTranspose(128, [3, 3], strides=2, activation='relu', padding='same')
        self.layer_2 = keras.layers.Conv2DTranspose(64, [3, 3], strides=2, activation='relu', padding='same')
        self.layer_3 = keras.layers.Conv2DTranspose(32, [3, 3], strides=2, activation='relu', padding='same')
        self.decoder_output = keras.layers.Conv2D(
            1, [3, 3], strides=1, activation='sigmoid', padding='same', name='decoder_output'
        )

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.decoder_output(x)


def create_autoencoder():

    encoder_input = keras.layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(encoder_input)
    x = keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
    shape_before_flattening = K.int_shape(x)[1:]
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(2, name="encoder_output")(x)
    encoder = keras.models.Model(encoder_input, encoder_output)

    decoder_input = keras.layers.Input(shape=(2, ), name='decoder_input')
    x = keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = keras.layers.Reshape(target_shape=shape_before_flattening)(x)
    x = keras.layers.Conv2DTranspose(128, [3, 3], strides=2, activation='relu', padding='same')(x)
    x = keras.layers.Conv2DTranspose(64, [3, 3], strides=2, activation='relu', padding='same')(x)
    x = keras.layers.Conv2DTranspose(32, [3, 3], strides=2, activation='relu', padding='same')(x)
    decoder_output = keras.layers.Conv2D(
        1, [3, 3], strides=1, activation='sigmoid', padding='same', name='decoder_output'
    )(x)
    decoder = keras.models.Model(decoder_input, decoder_output)

    autoencoder = keras.models.Model(encoder_input, decoder(encoder_output))

    return encoder, decoder, autoencoder


def create_decoder(shape_before_flattening):
    decoder_input = keras.layers.Input(shape=(2, ), name='decoder_input')
    x = keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = keras.layers.Reshape(target_shape=shape_before_flattening)(x)
    x = keras.layers.Conv2DTranspose(128, [3, 3], strides=2, activation='relu', padding='same')(x)
    x = keras.layers.Conv2DTranspose(64, [3, 3], strides=2, activation='relu', padding='same')(x)
    x = keras.layers.Conv2DTranspose(32, [3, 3], strides=2, activation='relu', padding='same')(x)
    decoder_output = keras.layers.Conv2D(
        1, [3, 3], strides=1, activation='sigmoid', padding='same', name='decoder_output'
    )(x)
    decoder = keras.models.Model(decoder_input, decoder_output)
    return decoder


#####################################################################################


# VARIATIONAL AUTOENCODER


class Sampling(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        sample = K.random_normal(shape=[tf.shape(z_mean)[0], tf.shape(z_mean)[1]])  # to use with reparameterization trick
        return z_mean + tf.exp(1/2 * z_log_var) * sample


def create_variational_encoder():
    encoder_input = keras.layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(encoder_input)
    x = keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
    shape_before_flattening = K.int_shape(x)[1:]
    x = keras.layers.Flatten()(x)
    z_mean = keras.layers.Dense(units=2, activation='linear', name='z_mean')(x)
    z_log_var = keras.layers.Dense(units=2, activation='linear', name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.models.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    return encoder, shape_before_flattening


class VAE(keras.models.Model):

    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                500 * keras.losses.binary_crossentropy(data, reconstruction, axis=[1, 2, 3])
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -1/2 * (1 + tf.exp(z_log_var) - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
                )
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}















































































































































































































