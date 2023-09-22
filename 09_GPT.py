import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import pandas as pd


WINE_PATH = '/home/talon/datasets/wine-reviews/winemag-data-130k-v2.json'
BATCH_SIZE = 2**5
VOCAB_SIZE = 10_000
MAX_LEN = 80


# PREP AND TOKENIZE THE DATA
def get_wine_og():
    def pad_punctuation(s):
        import re
        import string
        s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
        s = re.sub(" +", " ", s)
        return s
    with open(WINE_PATH) as json_data:  # context manager
        wine_data = json.load(json_data)
    filtered_data = [
        "wine review : "
        + x["country"]
        + " : "
        + x["province"]
        + " : "
        + x["variety"]
        + " : "
        + x["description"]
        for x in wine_data
        if x["country"] is not None
        and x["province"] is not None
        and x["variety"] is not None
        and x["description"] is not None
    ]
    text_data = [pad_punctuation(x) for x in filtered_data]
    text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(BATCH_SIZE).shuffle(1_000)
    # create vectorization layer
    vectorize_layer = keras.layers.TextVectorization(
        standardize='lower',
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_LEN + 1
    )
    # adapt layer to training set
    vectorize_layer.adapt(text_ds)
    vocabulary = vectorize_layer.get_vocabulary()
    return text_data, text_ds, vectorize_layer, vocabulary


def get_wine():
    # wine_csv = pd.read_csv(WINE_PATH[:-5] + '.csv')
    wine_json = pd.read_json(WINE_PATH)
    wine_json = wine_json.dropna(subset=['country', 'province', 'variety', 'description'])
    wines = wine_json.to_numpy()
    return wines


text_data, text_ds, vectorize_layer, vocab = get_wine_og()


























