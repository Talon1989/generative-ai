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


train_data = keras.utils.image_dataset_from_directory(
    '../global_data/lego-brick-images/dataset/', labels=None, color_mode='grayscale',
    image_size=[64, 64], batch_size=2**7, shuffle=True, seed=42, interpolation='bilinear'
)


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


discriminator, generator = Discriminator(), Generator()
dcgan = DCGAN(discriminator=discriminator, generator=generator)
dcgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000),
    g_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000)
)
# dcgan.fit(train, epochs=300)


wgan = WGAN_GP(discriminator=WganDiscriminator(), generator=Generator())












































































































































