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
from transormators import covariance_matrix
import matplotlib.pyplot as plt
import os
import glob


def preprocess(img):
    return tf.cast(img, 'float32') / 255.


encoder, shape_before_flattening = create_variational_encoder()
decoder = create_decoder(shape_before_flattening)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1/1_000))

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    mode="min",
    save_best_only=True,
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








