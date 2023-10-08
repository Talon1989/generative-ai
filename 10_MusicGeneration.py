import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import pandas as pd
import music21


# MONOPHONIC


BATCH_SIZE = 64


file = 'data/music/bach_cello/cs1-2all.mid'
example_score = music21.converter.parse(file).chordify()


def create_dataset(elements):
    ds = tf.data.Dataset.from_tensor_slices(elements)\
        .batch(BATCH_SIZE, drop_remainder=True).shuffle(1_000)
    vectorize_layer = keras.layers.TextVectorization(
        standardize=None, output_mode='int')
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()
    return ds, vectorize_layer, vocab













