import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
import pandas as pd
from call_me import SaveModel


WINE_PATH = '/home/fabio/.kaggle/datasets/wine-reviews/winemag-data-130k-v2.json'
PATH = '/home/talon/PycharmProjects/generative-ai/data/models/'
BATCH_SIZE = 2**5
VOCAB_SIZE = 10_000
MAX_LEN = 80
K_DIM = 2**7
V_DIM = 2**6
N_HEADS = 2
EMBEDDING_DIM = 2**8
F_F_DIM = 2**8


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


# CREATE TRAINING SET
def prep_inputs(text, vect_layer):
    text = tf.expand_dims(text, -1)
    tokenized_sentence = vect_layer(text)
    x = tokenized_sentence[:, :-1]
    y = tokenized_sentence[:, 1:]
    return x, y


train_ds = text_ds.map(lambda x: prep_inputs(x, vectorize_layer))


mha_layer = keras.layers.MultiHeadAttention(
    num_heads=N_HEADS,
    key_dim=K_DIM,
    value_dim=V_DIM,
    output_shape=256
)


# BUILDING THE TRANSFORMER


def attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]  # returns arrangement vertical
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.reshape(tf.cast(m, dtype), [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0
    )
    return tf.tile(mask, mult)


# att_mask = attention_mask(1, 10, 10, dtype=tf.int32)
# print(att_mask[0])


class TransformerBlock(keras.layers.Layer):
    def __init__(self, n_heads, k_dim, emb_dim, ff_dim, dropout_rate=1/10):
        super().__init__()
        self.n_heads = n_heads
        self.k_dim = k_dim
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.k_dim,
            output_shape=self.emb_dim
        )
        self.dropout_1 = keras.layers.Dropout(self.dropout_rate)
        self.l_n_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = keras.layers.Dense(units=self.ff_dim, activation='relu')
        self.ffn_2 = keras.layers.Dense(units=self.emb_dim)
        self.dropout_2 = keras.layers.Dropout(self.dropout_rate)
        self.l_n_2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = attention_mask(
            batch_size=batch_size, n_dest=seq_len, n_src=seq_len, dtype=tf.bool
        )  # hide future keys from query
        attention_output, attention_scores = self.mha(
            query=inputs,
            value=inputs,
            attention_mask=causal_mask,
            return_attention_scores=True
        )
        attention_output = self.dropout_1(attention_output)
        out_1 = self.l_n_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out_1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return self.l_n_2(out_1 + ffn_output), attention_scores


# non sinusoidal positional encoding
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, vocab_size, emb_dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.token_emb = keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.emb_dim
        )
        self.pos_emb = keras.layers.Embedding(
            input_dim=max_len, output_dim=emb_dim
        )

    def call(self, x):
        x_tokens = self.token_emb(x)
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        return x_tokens + positions


def make_gpt():
    inputs_ = keras.layers.Input(shape=(None, ), dtype=tf.int32)
    x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs_)
    x, attention_scores = TransformerBlock(
        N_HEADS, K_DIM, EMBEDDING_DIM, F_F_DIM
    )(x)
    outputs = keras.layers.Dense(units=VOCAB_SIZE, activation='softmax')(x)
    model = keras.models.Model(inputs_, [outputs, attention_scores])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1/1_000),
        loss=[keras.losses.SparseCategoricalCrossentropy(), None]
    )
    return model


gpt = make_gpt()


# save_callback = SaveModel(model=gpt, path='/home/talon/PycharmProjects/generative-ai/data/models/GPT')
# gpt.fit(train_ds, epochs=5, callbacks=[save_callback])





