import json
import re
import string
import numpy as np
import tensorflow as tf

keras = tf.keras
import pandas as pd
from utilities import *


# dataframe_recipe_data = pd.read_json('../global_data/epirecipes/full_format_recipes.json')
# recipe_data = dataframe_recipe_data.to_dict('records')


with open("../global_data/epirecipes/full_format_recipes.json") as json_data:
    recipe_data = json.load(json_data)
filtered_data = [
    "Recipe for " + x["title"] + "| " + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]


#  TOKENIZATION (WORD BASED AND NON STEMMED)


def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

# # turning text_data into a tf.Dataset to sample batches with
# text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1_000)
# # create vectorization layer
# vectorize_layer = keras.layers.TextVectorization(
#     standardize="lower",  # convert text to lowercase
#     max_tokens=10_000,  # gives token to most prevalent 10_000 words
#     output_mode="int",  # token is in integer form
#     output_sequence_length=200 + 1,  # trim or pad sequence to 201 token
# )
# # takes a dataset of tokens and computes vocabulary of string terms
# vectorize_layer.adapt(text_ds)
# vocab = vectorize_layer.get_vocabulary()
# # PREPPING INPUT
# def prepare_inputs(text):
#     text = tf.expand_dims(text, -1)
#     tokenized_sentences = vectorize_layer(text)
#     return tokenized_sentences[:, :-1], tokenized_sentences[:, 1:]
# train_ds = text_ds.map(prepare_inputs)


train_ds, vocabulary = tokenize_and_prep(text_data)


# LSTM


DICTIONARY_SIZE = 10_000
EMBEDDED_VECTOR_DIM = 100


class CustomLstm(keras.models.Model):
    def __init__(self):
        super(CustomLstm, self).__init__()
        # self.embedding = keras.layers.Embedding(
        #     input_dim=DICTIONARY_SIZE, output_dim=EMBEDDED_VECTOR_DIM, input_shape=(None,))
        self.embedding = keras.layers.Embedding(
            input_dim=DICTIONARY_SIZE, output_dim=EMBEDDED_VECTOR_DIM
        )
        self.lstm = keras.layers.LSTM(units=128, return_sequences=True)
        self.outputs = keras.layers.Dense(units=DICTIONARY_SIZE, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.outputs(x, training=training)
        return x


# inputs = keras.layers.Input(shape=(None, ), dtype='int32')
# x = keras.layers.Embedding(input_dim=DICTIONARY_SIZE, output_dim=EMBEDDED_VECTOR_DIM)(inputs)
# x = keras.layers.LSTM(units=128, return_sequences=True)(x)
# outputs = keras.layers.Dense(units=DICTIONARY_SIZE, activation='softmax')(x)
# lstm_1 = keras.models.Model(inputs, outputs)

lstm = CustomLstm()
lstm.build(
    input_shape=(  # specifies 3D input
        None,
        None,
    )
)
lstm.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy()
)


# CALLBACKS


class TextGenerator(keras.callbacks.Callback):

    def __init__(self, index_to_word: list):
        super(TextGenerator, self).__init__()
        self.index_to_word = index_to_word[:]
        self.word_to_index = {  # inverse word vocabulary (word -> token)
            word: index for index, word in enumerate(self.index_to_word)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        # words are first converted to tokens, then fed to beginning of generation
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            # outputs probas of each word (tokenized) of being the next word
            y = self.model.predict(x, verbose=0)
            # probas are passed through the sampler to sample the next word
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)  # append next word for next iteration
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/lstm_checkpoint",
    save_weights_only=False,
    save_freq='epoch',
    verbose=1
)


# TRAINING


lstm.fit(
    train_ds, epochs=25, verbose=1,
    callbacks=[model_checkpoint_callback, TextGenerator(vocabulary)]
)
lstm.save('./data/lstm_models/lstm')





















































