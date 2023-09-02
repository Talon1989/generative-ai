import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
keras = tf.keras
import keras.backend as K
from GanModels import Discriminator
from GanModels import Generator
from GanModels import DCGAN
from GanModels import WganDiscriminator
from GanModels import WGAN_GP
from GanModels import CGANGenerator
from GanModels import CGANDiscriminator
from utilities import display_images


Z_DIM = 128
N_FILES = 40_000
BATCH_SIZE = 2**7
# N_EPOCHS = 200
N_EPOCHS = 5


train_data = keras.utils.image_dataset_from_directory(
    '../global_data/lego-brick-images/dataset/',
    labels=None,
    color_mode='grayscale',  # 1 channel
    image_size=[64, 64],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation='bilinear'
)
# shuffle and repeat the dataset
train_data = train_data.shuffle(buffer_size=1000).repeat()


## TO USE DATASET FOR SUPERVISED LEARNING:
# X_train_g = tf.data.Dataset.from_tensor_slices(
#     np.expand_dims(X_train, axis=3)
# ).shuffle(1000).batch(param.BATCH_SIZE)


def preprocess(img):  # reprocessing the image to get values in range [-1, 1] such that we can use tanh
    mean = 127.5
    return (tf.cast(img, 'float32') - mean) / mean


train = train_data.map(lambda x: preprocess(x))


# # TO USE TO RETRIEVE NUMPY DATA FROM TF.DATASET OBJECT
# dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [5, 6]]))
# np_data = np.array([d for d in tfds.as_numpy(dataset)])


# #  TESTING
# discriminator = Discriminator()
# discriminator.compile(optimizer=keras.optimizers.Adam(1/1_000))
# test_train = np.asarray(list(train.unbatch()))
# #  saving small batches of test_train for testing
# np.save(file='data/test_train_8_batch.npy', arr=test_train[0:8])
# generator = Generator()
# discriminator.compile(optimizer=keras.optimizers.Adam(1/1_000))
# random_latent_vectors = tf.random.normal(shape=[3, 100], mean=0., stddev=1.)


# #  GAN
# discriminator, generator = Discriminator(), Generator()
# dcgan = DCGAN(discriminator=discriminator, generator=generator)
# dcgan.compile(
#     d_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000),
#     g_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000)
# )
# dcgan.fit(train, epochs=300)


#  WGAN


wgan = WGAN_GP(discriminator=WganDiscriminator(),
               generator=Generator(z_dim=(Z_DIM, )), latent_dim=Z_DIM)
wgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=1/5_000,
                                      beta_1=1/2, beta_2=999/1_000),
    g_optimizer=keras.optimizers.Adam(learning_rate=1/5_000,
                                      beta_1=1/2, beta_2=999/1_000)
)
# wgan_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath="./data/wgan_checkpoint.ckpt",
#     save_weights_only=True,
#     save_freq='epoch',
#     verbose=1,
# )


# CALLBACKS


wgan_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/wgan_checkpoint.ckpt",
    save_weights_only=True,
    save_freq='epoch',
    verbose=1,
)
wgan_tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./data/logs")


class ImageGenerator(keras.callbacks.Callback):

    def __init__(self, n_img=10, latent_dim=Z_DIM, display_frequency=1):
        super(ImageGenerator, self).__init__()
        self.n_img = n_img
        self.latent_dim = latent_dim
        self.display_frequency = display_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.display_frequency == 0:
            gen_img = self.model.generator(
                tf.random.normal(shape=[self.n_img, self.latent_dim])
            )
            gen_img = (gen_img * 127.5) + 127.5
            display_images(images=gen_img.numpy(), n=self.n_img)


# FITTING


steps_per_epoch = N_FILES // BATCH_SIZE
wgan.fit(train, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch,
         callbacks=[wgan_checkpoint_callback, wgan_tensorboard_callback, ImageGenerator(n_img=2, display_frequency=1)])
wgan.generator.save('./data/gan_models/wgan_generator')
wgan.discriminator.save('./data/gan_models/wgan_discriminator')


model = keras.models.load_model('./data/gan_models/wgan_generator')
z_sample = np.random.normal(loc=0., scale=1., size=[10, Z_DIM])
imgs = model.predict(z_sample)
display_images(imgs)
