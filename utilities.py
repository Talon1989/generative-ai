import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
keras = tf.keras

def covariance_matrix(matrix: np.array):
    mean_vector = np.mean(matrix, axis=0)
    mean_matrix = np.subtract(matrix, mean_vector)
    relation_matrix = np.dot(mean_matrix.T, mean_matrix)
    return relation_matrix / (matrix.shape[0] - 1)


def correlation_matrix(matrix: np.array):
    mean_vector = np.mean(matrix, axis=0)
    std_vector = np.std(matrix, axis=0)
    normalized_matrix = np.divide(np.subtract(matrix, mean_vector), std_vector)
    relation_matrix = np.dot(normalized_matrix.T, normalized_matrix)
    return relation_matrix / matrix.shape[0]


def display_images(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32"
):
    if images.max() > 1.0:  # normalizing the data
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0
    plt.figure(figsize=size)  # plotting
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")
    plt.show()
    plt.clf()


def tokenize_and_prep(text_data: list):

    def prep_inputs(text):
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        return tokenized_sentences[:, :-1], tokenized_sentences[:, 1:]

    vectorize_layer = keras.layers.TextVectorization(
        standardize="lower",  # convert text to lowercase
        max_tokens=10_000,  # gives token to most prevalent 10_000 words
        output_mode="int",  # token is in integer form
        output_sequence_length=200 + 1,  # trim or pad sequence to 201 token
    )
    vectorize_layer.adapt(text_data)
    text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1_000)
    return text_ds.map(prep_inputs), vectorize_layer.get_vocabulary()
    # vectorize_layer.adapt(text_ds)


# test commit on linux

