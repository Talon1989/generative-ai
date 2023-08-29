import numpy as np
import tensorflow as tf
from keras.backend import ones_like
from mkl_random import rand

keras = tf.keras


class Discriminator(keras.models.Model):

    def __init__(self, image_size=(64, 64, 1, )):
        super().__init__()
        self.input_layer = keras.layers.Input(shape=image_size)
        # self.layer_1 = keras.layers.Conv2D(
        #     filters=64, kernel_size=[4, 4], strides=2, padding='same', use_bias=False
        # )
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
        # x = self.input_layer(inputs)
        # x = self.layer_1(x, training=training)
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

    def __init__(self, z_dim=(100, )):
        super().__init__()
        #  input shape is a 100 element vector sampled from multivariate standard normal distribution
        # self.reshape = keras.layers.Reshape(target_shape=[1, 1, z_dim])
        self.reshape = keras.layers.Reshape(target_shape=[1, 1, z_dim[0]], input_shape=z_dim)
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
        # x = self.input_layer(inputs)
        # x = self.reshape(x)
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
            #  label smoothing to tame the discriminator
            real_noisy_labels = real_labels + 1/10 * tf.random.uniform(shape=tf.shape(real_preds))
            fake_labels = tf.zeros_like(fake_preds)
            #  label smoothing to tame the discriminator
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


################### WGAN ###################


class WganDiscriminator(Discriminator):

    def __init__(self, image_size=(64, 64, 1)):
        super().__init__(image_size)
        #  range of wgan discriminator is [-inf, +inf] (then bounded by L-constraint)
        self.layer_5 = keras.layers.Conv2D(1, kernel_size=[4, 4], strides=1, padding='valid', use_bias=False, activation='linear')

    #  batch normalization shouldn't be used on WGAN discriminator since it creates correlation
    #  between images in the same batch, which makes the gradient penalty loss less effective
    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.layer_1(x, training=training)
        x = self.leaky_relu_1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_2(x, training=training)
        x = self.leaky_relu_2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_3(x, training=training)
        x = self.leaky_relu_3(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_4(x, training=training)
        x = self.leaky_relu_4(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_5(x, training=training)
        x = self.flatten(x)
        return x


#  Wasserstein Generative Adversarial Network with Gradient Penalty
class WGAN_GP(keras.models.Model):

    def __init__(self, discriminator, generator, latent_dim=100, d_steps=3, gp_weight=10.):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss_metric = keras.metrics.Mean(name="d_wass_loss")
        self.d_gp_metric = keras.metrics.Mean(name="d_gp")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.d_wass_loss_metric,
            self.d_gp_metric,
            self.d_loss_metric,
            self.g_loss_metric
        ]

    #  Alternative way to enforce L-constraint than np.clip:
    #  The idea behind gradient_penalty method is that we get a noised difference(average)
    #  of the real and generated images, we check how the discriminator changes when we input those,
    #  we get a norm value that is proportional to sum of those, then we return
    #  the mean of the squared distance between that and L-constraint.
    #  In a nutshell we want to 'see' how 'far' from the L-constraint on average
    #  the discriminator gradient is when we deal with those images (real and generated)
    def gradient_penalty(self, batch_size, real_data, fake_data, l_constraint=1.):
        alpha = tf.random.normal(shape=[batch_size, 1, 1, 1], mean=0., stddev=1.)
        diff = fake_data - real_data
        interpolated_data = real_data + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_data)  # calculate gradient w respect to interpolated_data
            predictions = self.discriminator(interpolated_data, training=True)
        grads = gp_tape.gradient(predictions, [interpolated_data])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))  # l2
        grad_penalty = tf.reduce_mean((norm - l_constraint) ** 2)  # avg square distance between l2 and L-constraint
        return grad_penalty

    #  Unlike vanilla GAN, the discriminator must be trained to convergence before updating the generator,
    #  to ensure that the gradients for the generator update are accurate
    def train_step(self, data):

        batch_size = tf.shape(data)[0]
        for _ in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=[batch_size, self.latent_dim])
            with tf.GradientTape() as d_tape:
                fake_data = self.generator(random_latent_vectors, training=False)
                fake_predictions = self.discriminator(fake_data, training=True)
                real_predictions = self.discriminator(data, training=True)
                d_w_loss = tf.reduce_mean(fake_predictions - real_predictions)  # value is increasingly negative as d is correct
                d_gp = self.gradient_penalty(batch_size, data, fake_data)
                d_loss = d_w_loss + d_gp * self.gp_weight
            d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=[batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            fake_data = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.discriminator(fake_data, training=False)
            g_loss = tf.reduce_mean(fake_predictions)
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_w_loss)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}


################### CGAN ###################


class CGANDiscriminator(Discriminator):
    def __init__(self, image_size=(64, 64, 3, ), label_size=(64, 64, 2, )):
        super().__init__(image_size)
        self.concatenate = keras.layers.Concatenate(axis=-1, input_shape=[image_size, label_size])
        self.layer_1 = keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2, padding='same', use_bias=False)
        self.layer_5 = keras.layers.Conv2D(1, kernel_size=[4, 4], strides=1, padding='valid', use_bias=False, activation='linear')

    def call(self, inputs, training=False):
        x = self.concatenate([inputs[0], inputs[1]])
        x = self.layer_1(x, training=training)
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


#  hardcoded to work with 3 (rgb) channels
class CGANGenerator(Generator):

    def __init__(self, z_dim=(100, ), label_dim=(2, )):
        super().__init__(z_dim)
        self.concatenate = keras.layers.Concatenate(axis=-1, input_shape=[z_dim, label_dim])
        self.reshape = keras.layers.Reshape(target_shape=[1, 1, z_dim[0] + label_dim[0]])
        self.layer_5 = keras.layers.Conv2DTranspose(3, kernel_size=[4, 4], strides=2, padding='same', use_bias=False, activation='tanh')

    def call(self, inputs, training=False):
        x = self.concatenate([inputs[0], inputs[1]])
        x = self.reshape(x)
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


#  Conditional Wasserstein Generative Adversarial Network with Gradient Penalty
class CGAN(keras.models.Model):

    def __init__(self, discriminator, generator, latent_dim=100, d_steps=3, gp_weight=10.):
        super(CGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer):
        super(CGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss_metric = keras.metrics.Mean(name="d_wass_loss")
        self.d_gp_metric = keras.metrics.Mean(name="d_gp")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.d_wass_loss_metric,
            self.d_gp_metric,
            self.d_loss_metric,
            self.g_loss_metric
        ]

    def gradient_penalty(self, batch_size, real_data, fake_data, data_one_hot_label, l_constraint=1.):
        alpha = tf.random.normal(shape=[batch_size, 1, 1, 1], mean=0., stddev=1.)
        diff = fake_data - real_data
        interpolated_data = real_data + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_data)  # calculate gradient w respect to interpolated_data
            predictions = self.discriminator([interpolated_data, data_one_hot_label], training=True)
        grads = gp_tape.gradient(predictions, [interpolated_data])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))  # l2
        grad_penalty = tf.reduce_mean((norm - l_constraint) ** 2)  # avg square distance between l2 and L-constraint
        return grad_penalty

    def train_step(self, data):

        real_data, one_hot_labels = data

        data_one_hot_labels = one_hot_labels[:, None, None, :]
        data_one_hot_labels = tf.repeat(data_one_hot_labels, repeats=64, axis=1)
        data_one_hot_labels = tf.repeat(data_one_hot_labels, repeats=64, axis=2)
        batch_size = tf.shape(real_data)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=[batch_size, self.latent_dim])
            with tf.GradientTape() as d_tape:
                fake_data = self.generator([random_latent_vectors, one_hot_labels], training=False)
                fake_preds = self.discriminator([fake_data, data_one_hot_labels], training=True)
                real_preds = self.discriminator([real_data, data_one_hot_labels], training=True)
                d_w_loss = tf.reduce_mean(fake_preds - real_preds)
                d_gp = self.gradient_penalty(
                    batch_size, real_data, fake_data, data_one_hot_labels
                )
                d_loss = d_w_loss + self.gp_weight * d_gp
            d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=[batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([random_latent_vectors, one_hot_labels], training=True)
            fake_predictions = self.discriminator([fake_data, data_one_hot_labels], training=False)
            g_loss = tf.reduce_mean(fake_predictions)
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_w_loss)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}








