import matplotlib.pyplot as plt
import numpy as np


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


def display_generated_images(images, n=10, size=(20, 3), cmap='gray_r', as_type='float32'):
    if images.max() > 1.:  # normalizing the data
        images = images / 255.
    elif images.min() < 0.:
        images = (images + 1.) / 2.
    plt.figure(figsize=size)  # plotting
    for i in range(n):
        _ = plt.subplot(1, n, i+1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis('off')
    plt.show()
    plt.clf()
