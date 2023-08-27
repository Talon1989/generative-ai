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
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_3 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_4 = keras.layers.LeakyReLU(alpha=1/5)
        self.dropout = keras.layers.Dropout(rate=3/10)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs, training=False):
        x = self.layer_1(inputs, training=training)
        x = self.leaky_relu_1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_2(x, training=training)
        x = self.batch_norm_1(x, training=training)
        x = self.leaky_relu_2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_3(x, training=training)
        x = self.batch_norm_2(x, training=training)
        x = self.leaky_relu_3(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_4(x, training=training)
        x = self.batch_norm_3(x, training=training)
        x = self.leaky_relu_4(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_5(x, training=training)
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
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_3 = keras.layers.LeakyReLU(alpha=1/5)
        self.leaky_relu_4 = keras.layers.LeakyReLU(alpha=1/5)

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.layer_1(x, training=training)
        x = self.batch_norm_1(x, training=training)
        x = self.leaky_relu_1(x, training=training)
        x = self.layer_2(x, training=training)
        x = self.batch_norm_2(x, training=training)
        x = self.leaky_relu_2(x, training=training)
        x = self.layer_3(x, training=training)
        x = self.batch_norm_3(x, training=training)
        x = self.leaky_relu_3(x, training=training)
        x = self.layer_4(x, training=training)
        x = self.batch_norm_4(x, training=training)
        x = self.leaky_relu_4(x, training=training)
        x = self.layer_5(x, training=training)
        return x


#  vanilla Deep Convolutional Generative Adversarial Network
class DCGAN(keras.models.Model):

    def __init__(self, discriminator, generator, latent_dim=100):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_function = keras.losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name='discriminator_loss')
        self.g_loss_metric = keras.metrics.Mean(name='generator_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):

        batch_size = tf.shape(data)[0]
        random_latent_vectors = tf.random.normal(shape=[batch_size, self.latent_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_data = self.generator(random_latent_vectors, training=True)
            real_preds = self.discriminator(data, training=True)
            fake_preds = self.discriminator(generated_data, training=True)
            real_labels = tf.ones_like(real_preds)
            real_noisy_labels = real_labels + 1/10 * tf.random.uniform(shape=tf.shape(real_preds))
            fake_labels = tf.zeros_like(fake_preds)
            fake_noisy_labels = fake_labels + 1/10 * tf.random.uniform(shape=tf.shape(fake_preds))
            d_real_loss = self.loss_function(real_noisy_labels, real_preds)
            d_fake_loss = self.loss_function(fake_noisy_labels, fake_preds)
            d_loss = (d_real_loss + d_fake_loss) / 2.
            g_loss = self.loss_function(real_labels, fake_preds)

        d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}






