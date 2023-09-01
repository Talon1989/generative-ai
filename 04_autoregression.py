import json
import re
import string
import numpy as np
import tensorflow as tf
keras = tf.keras
import pandas as pd


# dataframe_recipe_data = pd.read_json('../global_data/epirecipes/full_format_recipes.json')
# recipe_data = dataframe_recipe_data.to_dict('records')


with open('../global_data/epirecipes/full_format_recipes.json') as json_data:
    recipe_data = json.load(json_data)
filtered_data = [
    'Recipe for ' + x['title'] + '| ' + ' '.join(x['directions'])
    for x in recipe_data
    if 'title' in x and x['title'] is not None and
       'directions' in x and x['directions'] is not None
]


#  TOKENIZATION (WORD BASED AND NON STEMMED)


def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r' \1 ', s)
    s = re.sub(' +', ' ', s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

# turning text_data into a tf.Dataset to sample batches with
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1_000)

# create vectorization layer
vectorize_layer = keras.layers.TextVectorization(
    standardize='lower',  # convert text to lowercase
    max_tokens=10_000,  # gives token to most prevalent 10_000 words
    output_mode='int',  # token is in integer form
    output_sequence_length=200+1  # trim or pad sequence to 201 token
)
# takes a dataset of tokens and computes vocabulary of string terms
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()


def print_words_and_vector():
    print(text_data[5])
    print()
    print(vectorize_layer(text_data[5]).numpy())






















































































































































































































