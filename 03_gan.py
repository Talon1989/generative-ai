import numpy as np
import tensorflow as tf
keras = tf.keras
import keras.backend as K
import pandas as pd
from GanModels import CGANGenerator
from GanModels import CGANDiscriminator
from GanModels import CGAN


N_CLASSES = 2
BATCH_SIZE = 2**7
N_CHANNELS = 3
Z_DIM = 2**6


#  LABEL READING
attributes = pd.read_csv("/home/talon/datasets/celeba/list_attr_celeba.csv")
# print(attributes.columns)
# attributes.head()
labels = attributes['Blond_Hair'].to_list()
int_labels = [x if x == 1 else 0 for x in labels]
train_data = keras.utils.image_dataset_from_directory(
    '/home/talon/datasets/celeba/img_align_celeba',
    labels=int_labels,
    color_mode='rgb',  # 3 channels
    image_size=[64, 64],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation='bilinear'
)


def preprocess(img):  # reprocessing the image to get values in range [-1, 1] such that we can use tanh
    mean = 127.5
    return (tf.cast(img, 'float32') - mean) / mean


train = train_data.map(lambda x, y: (preprocess(x), tf.one_hot(y, depth=N_CLASSES)))


generator = CGANGenerator(z_dim=(Z_DIM, ), label_dim=(N_CLASSES, ))
discriminator = CGANDiscriminator(image_size=(64, 64, N_CHANNELS, ), label_size=(64, 64, N_CLASSES, ))
cgan = CGAN(discriminator=discriminator, generator=generator, latent_dim=Z_DIM)
cgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000),
    g_optimizer=keras.optimizers.Adam(learning_rate=1/5_000, beta_1=1/2, beta_2=999/1_000)
)
cgan_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/cgan_checkpoint.ckpt",
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)
cgan_tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./data/logs")
cgan.fit(
    train,
    epochs=500,
    steps_per_epoch=2,
    callbacks=[cgan_checkpoint_callback, cgan_tensorboard_callback]
)
cgan.generator.save('./data/gan_models/cgan_generator')


# data_arr = []
# for batch in train.as_numpy_iterator():
#     data_arr.append(batch)























































































