import numpy as np
import torch
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # this is to deal with a PyCharm - matplolib issue
# matplotlib.use('Agg')  # this is to deal with a PyCharm - matplolib issue
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


def one_hot_transformation(y: np.array) -> np.array:
    """
    :param y: label encoded 1D np.array
    :return:
    """
    assert y.shape[1] is None
    n_unique = len(np.unique(y))
    one_hot = np.zeros(shape=[y.shape[0], n_unique])
    for idx, val in enumerate(y):
        one_hot[idx, int(val)] = 1
    return one_hot


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


def display_images(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32"):
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


# in torch, we have images in shape (channel, pixel, pixel) so we need
# to move the shape around to get (pixel, pixel, channel) compatible with plt
def display_images_torch(
    images: torch.Tensor, n=10, size=(20, 3), cmap="gray_r", as_type="float32"
):
    if images.max() > 1.0:  # normalizing the data
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0
    plt.figure(figsize=size)  # plotting
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].permute(1, 2, 0), cmap=cmap)
        plt.axis("off")
    plt.show()
    plt.clf()


def display_image_torch(image: torch.Tensor, cmap="gray_r"):
    if image.max() > 1.0:  # normalizing the data
        image = image / 255.0
    elif image.min() < 0.0:
        image = (image + 1.0) / 2.0
    plt.imshow(image.permute(1, 2, 0), cmap=cmap)
    plt.axis("off")
    plt.show()
    plt.clf()


def plot_line(values: np.array, name: str):
    plt.plot(np.arange(values.shape[0]), values, c="b")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(name)
    plt.show()
    plt.clf()


def plot_lines(values: np.array, names):
    try:
        values.shape[1]
    except IndexError:
        print("data is not in the correct format")
        return
    colors = ["b", "r", "g", "k"]
    for i in range(values.shape[0]):
        plt.plot(
            np.arange(values.shape[1]),
            values[i],
            c=colors[i % values.shape[0]],
            label=names[i],
        )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.show()
    plt.clf()


def tokenize_and_prep(text_data: list):
    import tensorflow as tf
    keras = tf.keras
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


def plot_2D(x, y):
    import matplotlib.pyplot as plt
    assert len(x) == len(y)
    plt.scatter(x, y, s=1)
    plt.show()
    plt.clf()


class ReplayBuffer:
    def __init__(self, max_size=1_000):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.states_, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )

    def get_buffer_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards)
        return len(self.actions)

    def remember(self, s, a, r, s_, done):
        if len(self.states) > self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.states_[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards, self.states_, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )

    def get_buffer(
        self, batch_size, randomized=True, cleared=False, return_bracket=False
    ):
        assert batch_size <= self.max_size + 1
        indices = np.arange(self.get_buffer_size())
        if randomized:
            np.random.shuffle(indices)
        buffer_states = np.squeeze([self.states[i] for i in indices][0:batch_size])
        buffer_actions = [self.actions[i] for i in indices][0:batch_size]
        buffer_rewards = [self.rewards[i] for i in indices][0:batch_size]
        buffer_states_ = np.squeeze([self.states_[i] for i in indices][0:batch_size])
        buffer_dones = [self.dones[i] for i in indices][0:batch_size]
        if cleared:
            self.clear()
        if return_bracket:
            for i in range(batch_size):
                buffer_actions[i] = np.array(buffer_actions[i])
                buffer_rewards[i] = np.array([buffer_rewards[i]])
                buffer_dones[i] = np.array([buffer_dones[i]])
            return (
                np.array(buffer_states),
                np.array(buffer_actions),
                np.array(buffer_rewards),
                np.array(buffer_states_),
                np.array(buffer_dones),
            )
            # return tuple(np.array(buffer_states)), tuple(np.array(buffer_actions)), tuple(np.array(buffer_rewards)), tuple(np.array(buffer_states_)), tuple(np.array(buffer_dones))
        return (
            np.array(buffer_states),
            np.array(buffer_actions),
            np.array(buffer_rewards),
            np.array(buffer_states_),
            np.array(buffer_dones),
        )

    def __len__(self):
        return len(self.actions)


def display_graph(scores, avg_scores, ep):
    plt.scatter(np.arange(len(scores)), scores, c='g', s=1, label='scores')
    plt.plot(avg_scores, c='b', linewidth=1, label='avg scores')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.legend(loc='best')
    plt.title('Episode %d DQL' % ep)
    plt.show()
    plt.clf()
