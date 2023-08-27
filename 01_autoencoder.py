import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import keras.backend as K
from Autoencoders import Encoder
from Autoencoders import create_autoencoder
from Autoencoders import create_variational_encoder
from Autoencoders import create_decoder
from Autoencoders import VAE
from Autoencoders import create_variational_encoder_celeba
from Autoencoders import create_decoder_celeba
from transormators import covariance_matrix
import matplotlib.pyplot as plt
import os
import glob
from face_parameters import *


def preprocess(img):
    return tf.cast(img, 'float32') / 255.


def old_preprocess(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs, [[0, 0], [2, 2], [2, 2]], constant_values=0.)
    imgs = np.expand_dims(imgs, -1)
    return imgs


# encoder, shape_before_flattening = create_variational_encoder()
# decoder = create_decoder(shape_before_flattening)
# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1/1_000))

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/vae_faces_checkpoint",
    save_weights_only=False,
    # save_freq="epoch",
    save_freq=20,  # saving the checkpoint every 20 batches
    # monitor='total_loss_tracker',
    mode="min",
    # save_best_only=True,
    save_best_only=False,
    verbose=0,
)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

# vae.fit(
#     X_train, epochs=5, batch_size=100, shuffle=True, validation_data=[X_test, X_test],
#     callbacks=[model_checkpoint_callback, tensorboard_callback]
# )
# vae.save('./data/vae')
# encoder.save('./data/encoder')
# decoder.save('./data/decoder')

# vae.fit(
#     X_train, epochs=5, batch_size=100, shuffle=True
# )


PATH = '../global_data/CelebA/img_align_celeba/img_align_celeba'

train_data = keras.utils.image_dataset_from_directory(
    PATH, labels=None, color_mode='rgb', image_size=[64, 64], batch_size=128, shuffle=True, seed=42, interpolation='bilinear'
)
train = train_data.map(lambda x: preprocess(x))

# vae.fit(
#     train, epochs=5, batch_size=100, shuffle=True
# )
# vae.save('./data/celeba_vae')
# encoder.save('./data/celeba_encoder')
# decoder.save('./data/celeba_decoder')

encoder, shape_before_flattening = create_variational_encoder_celeba()
decoder = create_decoder_celeba(shape_before_flattening)
vae = VAE(encoder, decoder)  # passing a reference to that model object, changes to encoder and decoder are saved (like array obj)
vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
)

# vae.fit(
#     train, epochs=5, batch_size=train_data._batch_size, shuffle=True, callbacks=[model_checkpoint_callback]
# )
# vae.save('./data/celeba_vae')
# encoder.save('./data/celeba_encoder')
# decoder.save('./data/celeba_decoder')
#
#
# def face_generation():
#     grid_width, grid_height = 10, 3
#     image_size = grid_width * grid_height
#     z_sample = np.random.normal(loc=0, scale=1, size=[image_size, 200])
#     reconstructions = decoder.predict(z_sample)
#     fig = plt.figure(figsize=(18, 5))
#     fig.subplots_adjust(hspace=.4, wspace=.4)
#     for i in range(image_size):
#         ax = fig.add_subplot(grid_height, grid_width, i+1)
#         ax.axis('off')
#         ax.imshow(reconstructions[i, :, :])

















