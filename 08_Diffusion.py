import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from call_me import SaveModel
from call_me import DiffusionSaveModel
import warnings
keras = tf.keras
from utilities import *
from custom_layers import ResidualBlock, UpBlock, DownBlock
# warnings.filterwarnings('ignore')


PATH = "/home/talon/datasets/flower-dataset/dataset"
EMA = 999 / 1_000
NOISE_EMBEDDING_SIZE = 32


train_data = keras.utils.image_dataset_from_directory(
    directory=PATH,
    labels=None,
    image_size=[64, 64],
    batch_size=None,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)


# scale to range [0, 1]
def preprocess(img):
    return tf.cast(img, "float32") / 255.0


train = train_data.map(lambda x: preprocess(x))
train = train.repeat(5)  # repeat dataset 5 times
train = train.batch(2**6, drop_remainder=True)


def linear_diffusion_schedule(diffusion_times):
    min_rate = 1 / 10_000
    max_rate = 1 / 50
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
    min_signal_rate = 1 / 50
    max_signate_rate = 19 / 20
    start_angle = tf.acos(max_signate_rate)
    end_angle = tf.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)
    return noise_rates, signal_rates


T = 1_000
# diff_times = [x/T for x in range(T)]
# diff_times = tf.convert_to_tensor([x / T for x in range(T)])
# linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(diff_times)


class DiffusionModel(keras.models.Model):
    def __init__(self, model: keras.models.Model, diff_schedule):
        super().__init__()
        self.normalizer = keras.layers.Normalization()
        self.network = model
        self.ema_network = keras.models.clone_model(self.network)
        self.diffusion_schedule = diff_schedule
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")

    # def call(self, inputs, training=False):
    #     # Dummy call method. It just returns the inputs without any computation.
    #     return inputs

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
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
            shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0
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
        self.noise_loss_tracker.update_state(noise_loss)

        # EMA (soft) update
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        n_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):
            # set all diffusion times to1 (start of reverse diffusion process)
            diffusion_times = tf.ones(shape=(n_images, 1, 1, 1)) - step * step_size
            # noise and signal rates are calculated according to the diffusion schedule
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )  # unet predicts the noise and returns the denoised image estimate (step 1)
            # reduce diffusion times by one step
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )  # next noise and signal rates are calculated
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )  # t-1 images are calculated (step 2)
        return pred_images

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**1 / 2
        return tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=1.0)

    def generate(self, n_images, diffusion_steps):
        initial_noise = tf.random.normal(shape=[n_images, 64, 64, 3])
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        return self.denormalize(generated_images)


def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),  # start
            tf.math.log(1000.0),  # end
            NOISE_EMBEDDING_SIZE // 2,  # number of elements (dimensions)
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def make_unet():

    def ResidualBlock(width):
        def apply(x):
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = keras.layers.Conv2D(width, kernel_size=[1, 1])(x)
            x = keras.layers.BatchNormalization(center=False, scale=False)(x)
            x = keras.layers.Conv2D(
                width,
                kernel_size=[3, 3],
                padding="same",
                activation=keras.activations.swish,
            )(x)
            x = keras.layers.Conv2D(width, kernel_size=[3, 3], padding="same")(x)
            x = keras.layers.Add()([x, residual])
            return x

        return apply

    def DownBlock(width, block_depth):
        def apply(x):
            x, skips = x
            print(len(skips))
            for _ in range(block_depth):
                x = ResidualBlock(width)(x)
                skips.append(x)
            x = keras.layers.AveragePooling2D(pool_size=2)(x)
            return x

        return apply

    def UpBlock(width, block_depth):
        def apply(x):
            x, skips = x
            x = keras.layers.UpSampling2D(size=[2, 2], interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = keras.layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width)(x)
            return x

        return apply

    # image we wish to denoise
    noisy_images = keras.layers.Input(shape=[64, 64, 3])
    x = keras.layers.Conv2D(32, kernel_size=[1, 1])(noisy_images)

    # noise variance
    noise_variance = keras.layers.Input(shape=[1, 1, 1])
    noise_embedding = keras.layers.Lambda(sinusoidal_embedding)(noise_variance)
    noise_embedding = keras.layers.UpSampling2D(size=[64, 64], interpolation="nearest")(
        noise_embedding
    )

    x = keras.layers.Concatenate()([x, noise_embedding])

    # skips list hold the output from DownBlock layers
    # that we wish to connect to UpBlock layers downstream
    skips = []

    x = DownBlock(width=32, block_depth=2)([x, skips])
    x = DownBlock(width=64, block_depth=2)([x, skips])
    x = DownBlock(width=96, block_depth=2)([x, skips])

    x = ResidualBlock(width=128)(x)
    x = ResidualBlock(width=128)(x)

    x = UpBlock(width=96, block_depth=2)([x, skips])
    x = UpBlock(width=64, block_depth=2)([x, skips])
    x = UpBlock(width=32, block_depth=2)([x, skips])

    x = keras.layers.Conv2D(filters=3, kernel_size=[1, 1], kernel_initializer="zeros")(
        x
    )

    unet = keras.models.Model([noisy_images, noise_variance], x, name="unet")

    return unet


def make_unet_class_inheritance():
    # image we wish to denoise
    noisy_images = keras.layers.Input(shape=[64, 64, 3])
    x = keras.layers.Conv2D(32, kernel_size=[1, 1])(noisy_images)

    # noise variance
    noise_variance = keras.layers.Input(shape=[1, 1, 1])
    noise_embedding = keras.layers.Lambda(sinusoidal_embedding)(noise_variance)
    noise_embedding = keras.layers.UpSampling2D(size=[64, 64], interpolation="nearest")(
        noise_embedding
    )

    x = keras.layers.Concatenate()([x, noise_embedding])

    # skips list hold the output from DownBlock layers
    # that we wish to connect to UpBlock layers downstream
    skips = []

    x, skips = DownBlock(width=32, block_depth=2)([x, skips])
    x, skips = DownBlock(width=64, block_depth=2)([x, skips])
    x, skips = DownBlock(width=96, block_depth=2)([x, skips])

    x = ResidualBlock(width=128)(x)
    x = ResidualBlock(width=128)(x)

    x, skips = UpBlock(width=96, block_depth=2)([x, skips])
    x, skips = UpBlock(width=64, block_depth=2)([x, skips])
    x, skips = UpBlock(width=32, block_depth=2)([x, skips])

    x = keras.layers.Conv2D(filters=3, kernel_size=[1, 1], kernel_initializer="zeros")(
        x
    )

    unet = keras.models.Model([noisy_images, noise_variance], x, name="unet")

    return unet


class UNET(keras.models.Model):
    def __init__(self):
        super().__init__()

        self.image_conv = keras.layers.Conv2D(
            32, kernel_size=[1, 1], input_shape=[None, 64, 64, 3]
        )
        self.noise_embedding = keras.layers.Lambda(
            sinusoidal_embedding, input_shape=[None, 1, 1, 1]
        )
        self.noise_upsampling = keras.layers.UpSampling2D(
            size=[64, 64], interpolation="nearest"
        )
        self.skips = []
        self.down_block_1 = DownBlock(width=32, block_depth=2)
        self.down_block_2 = DownBlock(width=64, block_depth=2)
        self.down_block_3 = DownBlock(width=96, block_depth=2)
        self.concat = keras.layers.Concatenate()
        self.residual_block_1 = ResidualBlock(width=128)
        self.residual_block_2 = ResidualBlock(width=128)
        self.up_block_1 = UpBlock(width=96, block_depth=2)
        self.up_block_2 = UpBlock(width=64, block_depth=2)
        self.up_block_3 = UpBlock(width=32, block_depth=2)
        self.out_conv = keras.layers.Conv2D(
            filters=3, kernel_size=[1, 1], kernel_initializer="zeros"
        )

    def call(self, inputs, training=None, mask=None):
        noisy_images, noise_variance = inputs

        noisy_images = self.image_conv(noisy_images, training=training)
        noise_variance = self.noise_embedding(noise_variance)
        noise_variance = self.noise_upsampling(noise_variance)

        x = self.concat([noisy_images, noise_variance])
        skips = []

        x, skips = self.down_block_1([x, skips], training=training)
        x, skips = self.down_block_2([x, skips], training=training)
        x, skips = self.down_block_3([x, skips], training=training)

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        x, skips = self.up_block_1([x, skips], training=training)
        x, skips = self.up_block_2([x, skips], training=training)
        x, _ = self.up_block_3([x, skips], training=training)

        x = self.out_conv(x)

        return x


# unet = make_unet_class_inheritance()
unet = UNET()
unet.build(input_shape=[(None, 64, 64, 3), (None, 1, 1, 1)])


# # Example input tensor of shape (1, 2, 2, 1)
# x = tf.constant(
#     [
#         [
#             [[1.0], [2.0]],
#             [[3.0], [4.0]]
#         ]
#     ]
# )
# output = sinusoidal_embedding(x)
# print(output)


# model = DiffusionModel(model=unet, diff_schedule=cosine_diffusion_schedule)
# # save_model_callback = SaveModel(model.network, "data/models/U-Net")
# save_model_callback = DiffusionSaveModel(model, "data/models/U-Net")
# model.compile(
#     optimizer=keras.optimizers.experimental.AdamW(
#         learning_rate=1e-3, weight_decay=1e-4
#     ),
#     loss=keras.losses.mean_absolute_error,
# )
# model.normalizer.adapt(train)
# model.fit(train, epochs=10, callbacks=[save_model_callback])


model = DiffusionModel(
    model=keras.models.load_model('data/models/U-Net'),
    diff_schedule=cosine_diffusion_schedule
)
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=1e-3, weight_decay=1e-4
    ),
    loss=keras.losses.mean_absolute_error,
)
model.normalizer.adapt(train)
