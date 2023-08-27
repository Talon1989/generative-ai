import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import keras.backend as K
from GanModels import Discriminator
from GanModels import Generator


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


#  TESTING
discriminator = Discriminator()
discriminator.compile(optimizer=keras.optimizers.Adam(1/1_000))
test_train = np.asarray(list(train.unbatch()))
#  saving small batches of test_train for testing
np.save(file='data/test_train_8_batch.npy', arr=test_train[0:8])
generator = Generator()
discriminator.compile(optimizer=keras.optimizers.Adam(1/1_000))
random_latent_vectors = tf.random.normal(shape=[3, 100], mean=0., stddev=1.)
