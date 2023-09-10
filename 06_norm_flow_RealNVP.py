import json
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets
keras = tf.keras
from keras import layers, models, regularizers
import pandas as pd
from utilities import *
from sklearn import datasets
import os


PATH = "data/models/realNVP_00"


data = datasets.make_moons(n_samples=3_000, noise=1 / 20)[0].astype("float32")
norm = keras.layers.Normalization()
norm.adapt(data)
norm_data = norm(data)


plot_2D(norm_data.numpy()[:, 0], norm_data.numpy()[:, 1])


def coupling(input_dim=2):
    input_layer = layers.Input(shape=2)

    # Scale Layers
    s_layer_1 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(input_layer)
    s_layer_2 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(s_layer_1)
    s_layer_3 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(s_layer_2)
    s_layer_4 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(s_layer_3)
    s_layer_5 = layers.Dense(
        units=input_dim, activation="tanh", kernel_regularizer=regularizers.l2(1 / 100)
    )(s_layer_4)

    # Translation Layers
    t_layer_1 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(input_layer)
    t_layer_2 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(t_layer_1)
    t_layer_3 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(t_layer_2)
    t_layer_4 = layers.Dense(
        units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
    )(t_layer_3)
    t_layer_5 = layers.Dense(
        units=input_dim,
        activation="linear",
        kernel_regularizer=regularizers.l2(1 / 100),
    )(t_layer_4)

    return models.Model(input_layer, [s_layer_5, t_layer_5])


class Coupling(models.Model):
    def __init__(self, input_dim=2):
        super(Coupling, self).__init__()
        # Scale Layers
        self.s_layer_1 = layers.Dense(
            units=256,
            activation="relu",
            input_shape=(input_dim,),
            kernel_regularizer=regularizers.l2(1 / 100),
        )
        self.s_layer_2 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.s_layer_3 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.s_layer_4 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.s_layer_5 = layers.Dense(
            units=input_dim,
            activation="tanh",
            kernel_regularizer=regularizers.l2(1 / 100),
        )
        # Translation Layers
        self.t_layer_1 = layers.Dense(
            units=256,
            activation="relu",
            input_shape=(input_dim,),
            kernel_regularizer=regularizers.l2(1 / 100),
        )
        self.t_layer_2 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.t_layer_3 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.t_layer_4 = layers.Dense(
            units=256, activation="relu", kernel_regularizer=regularizers.l2(1 / 100)
        )
        self.t_layer_5 = layers.Dense(
            units=input_dim,
            activation="linear",
            kernel_regularizer=regularizers.l2(1 / 100),
        )

    def call(self, inputs, training=None, mask=None):
        s = self.s_layer_1(inputs, training=training)
        s = self.s_layer_2(s, training=training)
        s = self.s_layer_3(s, training=training)
        s = self.s_layer_4(s, training=training)
        s = self.s_layer_5(s, training=training)
        t = self.t_layer_1(inputs, training=training)
        t = self.t_layer_2(t, training=training)
        t = self.t_layer_3(t, training=training)
        t = self.t_layer_4(t, training=training)
        t = self.t_layer_5(t, training=training)
        return [s, t]


# couple = Coupling()
# couple.build(input_shape=(None, 2))
# print(couple.summary())  # ( (2+1)*256 + (256+1)*256*3 + (256+1)*2) * 2  =  397_316


class RealNVP(models.Model):
    def __init__(
        self, input_dim, coupling_layers, coupling_dim=256, regularization=1 / 100
    ):
        """
        :param input_dim:
        :param coupling_layers:
        :param coupling_dim: defaulted 256
        :param regularization: defaulted 1/100 (l2)
        """
        super().__init__()
        self.coupling_layers = coupling_layers
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )  # standard normal distribution
        self.masks = np.array(
            [[0, 1], [1, 0]] * (self.coupling_layers // 2), dtype="float32"
        )  # alternating mask pattern
        self.loss_tracker = keras.metrics.Mean("loss")
        self.layers_list = [
            Coupling(input_dim) for _ in range(self.coupling_layers)
        ]  # RealNVP

    # Need to call custom_arguments method of .load_model to run this
    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = cls(**config)
        # need to add this because tfp is not natively supported by tf keras .load_model
        model.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )
        return model

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=None, mask=None):
        log_det_inv, direction = 0, 1
        x = None
        if training:
            direction = -1
        for i in range(self.coupling_layers)[::direction]:
            x_masked = inputs * self.masks[i]
            reverse_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s = s * reverse_mask
            t = t * reverse_mask
            gate = (direction - 1) / 2
            # forward and backward equations dependent on the direction
            x = (
                reverse_mask * (inputs * tf.exp(s * direction))
                + (t * direction * tf.exp(s * gate))
            ) + x_masked
            # log determinant of jacobian
            log_det_inv = log_det_inv + (tf.reduce_sum(s, axis=1) * gate)
        return x, log_det_inv

    def log_loss(self, x):
        y, log_det = self(x)
        # p_x(x) = p_z(f(x)) * det(J) , then log
        log_likelihood = self.distribution.log_prob(y) + log_det
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


class ImageGenerator(keras.callbacks.Callback):

    def __init__(self, n_samples, data):
        super().__init__()
        self.n_samples = n_samples
        self.data = data

    def generate(self):
        z, _ = self.model(self.data)
        samples = self.model.distribution.sample(self.n_samples)
        x, _ = self.model.predict(samples, verbose=0)
        return x, z, samples

    def display(self, x, z, samples):
        f, axes = plt.subplots(2, 2)
        axes[0, 0].scatter(
            self.data[:, 0], self.data[:, 1], color="r", s=1
        )
        axes[0, 0].set(title="Data space X", xlabel="x_1", ylabel="x_2")
        axes[0, 0].set_xlim([-2, 2])
        axes[0, 0].set_ylim([-2, 2])
        axes[0, 1].scatter(z[:, 0], z[:, 1], color="r", s=1)
        axes[0, 1].set(title="f(X)", xlabel="z_1", ylabel="z_2")
        axes[0, 1].set_xlim([-2, 2])
        axes[0, 1].set_ylim([-2, 2])
        axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g", s=1)
        axes[1, 0].set(title="Latent space Z", xlabel="z_1", ylabel="z_2")
        axes[1, 0].set_xlim([-2, 2])
        axes[1, 0].set_ylim([-2, 2])
        axes[1, 1].scatter(x[:, 0], x[:, 1], color="g", s=1)
        axes[1, 1].set(title="g(Z)", xlabel="x_1", ylabel="x_2")
        axes[1, 1].set_xlim([-2, 2])
        axes[1, 1].set_ylim([-2, 2])
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        plt.show()
        plt.clf()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            x, z, samples = self.generate()
            self.display(x, z, samples)


if not os.path.exists(PATH):
    model = RealNVP(input_dim=2, coupling_layers=6)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1 / 100_000))
    model.fit(
        norm_data, batch_size=256, epochs=800
    )
    model.save('data/models/realNVP_00')
else:
    # need to call custom_objects argument to use from_config method
    model = keras.models.load_model(PATH, custom_objects={'RealNVP': RealNVP})

img_generator = ImageGenerator(n_samples=3_000, data=norm_data)
img_generator.set_model(model)
x, z, samples = img_generator.generate()
img_generator.display(x, z, samples)
