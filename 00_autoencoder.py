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


(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
train_indices = np.random.choice(X_train.shape[0], size=2_000, replace=False)
test_indices = np.random.choice(X_test.shape[0], size=1_000, replace=False)
X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]


def preprocess(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs, [[0, 0], [2, 2], [2, 2]], constant_values=0.)
    imgs = np.expand_dims(imgs, -1)
    return imgs


X_train = preprocess(X_train)
X_test = preprocess(X_test)


# encoder = Encoder()
# print(encoder(X_train[0:3]))


# if len(glob.glob('data/autoencoder*')) == 0:
#     encoder, decoder, autoencoder = create_autoencoder()
#     autoencoder.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=1 / 1_000),
#         loss=keras.losses.binary_crossentropy
#     )
#     autoencoder.fit(
#         X_train, X_train, epochs=500, batch_size=100, shuffle=True, validation_data=(X_test, X_test)
#     )
#     autoencoder.save(filepath='data/autoencoder_ep_500.h5')
# else:
#     autoencoder = keras.models.load_model('data/autoencoder_ep_500.h5')

# encoder_input = keras.layers.Input(shape=(32, 32, 1), name="encoder_input")
# x = keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(encoder_input)
# x = keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
# x = keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
# shape_before_flattening = K.int_shape(x)[1:]
# x = keras.layers.Flatten()(x)
# encoder_output = keras.layers.Dense(2, name="encoder_output")(x)
# encoder = keras.models.Model(encoder_input, encoder_output)
#
# decoder_input = keras.layers.Input(shape=(2,), name='decoder_input')
# x = keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
# x = keras.layers.Reshape(target_shape=shape_before_flattening)(x)
# x = keras.layers.Conv2DTranspose(128, [3, 3], strides=2, activation='relu', padding='same')(x)
# x = keras.layers.Conv2DTranspose(64, [3, 3], strides=2, activation='relu', padding='same')(x)
# x = keras.layers.Conv2DTranspose(32, [3, 3], strides=2, activation='relu', padding='same')(x)
# decoder_output = keras.layers.Conv2D(
#     1, [3, 3], strides=1, activation='sigmoid', padding='same', name='decoder_output'
# )(x)
# decoder = keras.models.Model(decoder_input, decoder_output)
#
# autoencoder = keras.models.Model(encoder_input, decoder(encoder_output))


def visualize_latent_space():
    embeddings = encoder.predict(X_test)
    plt.figure(figsize=(8, 8))
    colors = y_test
    # plt.scatter(embeddings[:, 0], embeddings[:, 1], c='black', alpha=1/2, s=3)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=3, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('labels')
    plt.show()


# custom_matrix = np.random.normal(loc=0, scale=15, size=[10, 4])
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

vae.fit(
    X_train, epochs=5, batch_size=100, shuffle=True, validation_data=[X_test, X_test],
    # callbacks=[model_checkpoint_callback, tensorboard_callback]
)
# vae.save('./data/vae')
# encoder.save('./data/encoder')
# decoder.save('./data/decoder')

# vae.fit(
#     X_train, epochs=5, batch_size=100, shuffle=True
# )



































































































































































