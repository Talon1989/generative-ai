import json
import re
import string
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets
keras = tf.keras
from keras import layers, models, regularizers
import pandas as pd
from utilities import *
from sklearn import datasets


data = datasets.make_moons(n_samples=3_000, noise=1/20)[0].astype('float32')
norm = keras.layers.Normalization()
norm.adapt(data)
norm_data = norm(data)


plot_2D(norm_data.numpy()[:, 0], norm_data.numpy()[:, 1])


def coupling(input_dim = 2):

    input_layer = layers.Input(shape=2)

    # Scale Layers
    s_layer_1 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(input_layer)
    s_layer_2 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(s_layer_1)
    s_layer_3 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(s_layer_2)
    s_layer_4 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(s_layer_3)
    s_layer_5 = layers.Dense(
        units=input_dim, activation='tanh',
        kernel_regularizer=regularizers.l2(1/100))(s_layer_4)

    # Translation Layers
    t_layer_1 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(input_layer)
    t_layer_2 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(t_layer_1)
    t_layer_3 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(t_layer_2)
    t_layer_4 = layers.Dense(
        units=256, activation='relu',
        kernel_regularizer=regularizers.l2(1/100))(t_layer_3)
    t_layer_5 = layers.Dense(
        units=input_dim, activation='linear',
        kernel_regularizer=regularizers.l2(1/100))(t_layer_4)

    return models.Model(input_layer, [s_layer_5, t_layer_5])


class Coupling(models.Model):

    def __init__(self, input_dim=2):
        super(Coupling, self).__init__()
        # Scale Layers
        self.s_layer_1 = layers.Dense(
            units=256, activation='relu', input_shape=(input_dim, ),
            kernel_regularizer=regularizers.l2(1 / 100))
        self.s_layer_2 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.s_layer_3 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.s_layer_4 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.s_layer_5 = layers.Dense(units=input_dim, activation='tanh',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        # Translation Layers
        self.t_layer_1 = layers.Dense(
            units=256, activation='relu', input_shape=(input_dim, ),
            kernel_regularizer=regularizers.l2(1 / 100))
        self.t_layer_2 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.t_layer_3 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.t_layer_4 = layers.Dense(units=256, activation='relu',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        self.t_layer_5 = layers.Dense(
            units=input_dim, activation='linear', kernel_regularizer=regularizers.l2(1 / 100))

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


class SimpleCoupling(models.Model):

    def __init__(self, input_dim=2):
        super(SimpleCoupling, self).__init__()
        # Scale Layers
        self.s_layer_1 = layers.Dense(
            units=256, activation='relu', name='s_1',
            kernel_regularizer=regularizers.l2(1 / 100))
        self.s_layer_2 = layers.Dense(units=input_dim, activation='tanh', name='s_2',
                                 kernel_regularizer=regularizers.l2(1 / 100))
        # Translation Layers
        self.t_layer_1 = layers.Dense(
            units=256, activation='relu', name='t_1',
            kernel_regularizer=regularizers.l2(1 / 100))
        self.t_layer_2 = layers.Dense(units=input_dim, activation='linear', name='t_2',
                                 kernel_regularizer=regularizers.l2(1 / 100))

    def call(self, inputs, training=None, mask=None):
        s = self.s_layer_1(inputs, training=training)
        s = self.s_layer_2(s, training=training)
        t = self.t_layer_1(inputs, training=training)
        t = self.t_layer_2(t, training=training)
        return [s, t]


couple = Coupling()
couple.build(input_shape=(None, 2))
print(couple.summary())  # ( (2+1)*256 + (256+1)*256*3 + (256+1)*2) * 2


class RealNVP(models.Model):

    def __init__(self, input_dim, coupling_layers, coupling_dim=256, regularization=1/100):
        """
        :param input_dim:
        :param coupling_layers:
        :param coupling_dim: defaulted 256
        :param regularization: defaulted 1/100 (l2)
        """
        super().__init__()
        self.coupling_layers = coupling_layers
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0., 0.], scale_diag=[1., 1.]
        )  # standard normal distribution
        self.masks = np.array(
            [[0, 1], [1, 0]] * (self.coupling_layers // 2), dtype='float32'
        )  # alternating mask pattern
        self.loss_tracker = keras.metrics.Mean('loss')
        self.layers_list = [
            Coupling(input_dim) for _ in range(self.coupling_layers)
        ]  # RealNVP

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
            x = (reverse_mask * (inputs * tf.exp(s * direction))
                 + (t * direction * tf.exp(s * gate))) + x_masked
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
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}


model = RealNVP(input_dim=2, coupling_layers=6)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1/10_000))
# model.fit(
#     norm_data, batch_size=256, epochs=300
# )





















































































































































































































































































