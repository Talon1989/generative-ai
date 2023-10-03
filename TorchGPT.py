import numpy as np
import copy
import time
import os
import json
import time
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchmetrics
import torchtext
from torch.utils.data.dataset import T_co

from custom_modules_torch import ResidualBlock, DownBlock, UpBlock
from utilities import display_images_torch, display_image_torch


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

    # text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(BATCH_SIZE).shuffle(1_000)
    # # create vectorization layer
    # vectorize_layer = keras.layers.TextVectorization(
    #     standardize='lower',
    #     max_tokens=VOCAB_SIZE,
    #     output_mode='int',
    #     output_sequence_length=MAX_LEN + 1
    # )
    # # adapt layer to training set
    # vectorize_layer.adapt(text_ds)
    # vocabulary = vectorize_layer.get_vocabulary()
    # return text_data, text_ds, vectorize_layer, vocabulary
    return text_data


text_data = get_wine_og()


# GPT3 APPROACH
# tokenizer = torchtext.data.utils.get_tokenizer('spacy')
# tokenized_data = [tokenizer(review) for review in text_data]
tokenized_data = [review.split() for review in text_data]
# set is used to ensure uniform spreading of key-words in the vocabulary
# (also set removes duplicates)
vocab = {word: idx for idx, word in enumerate(
    set(word for review in tokenized_data for word in review))}
vocab['<pad>'] = len(vocab)  # adding one more mapping for zero padding


class TextVectorization(Dataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self):
        super().__init__()


# BUILDING THE TRANSFORMER


def attention_mask(batch_size, n_dest, n_src, dtype):
    i = torch.arange(n_dest)[:, None]
    j = torch.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = torch.reshape(m.to(dtype), [1, n_dest, n_src])
    mult = torch.cat(
        [torch.unsqueeze(torch.tensor(batch_size), -1), torch.tensor([1, 1], dtype=dtype)],
        dim=0)
    return mask.repeat(tuple(mult.tolist()))


# mask = attention_mask(16, 10, 8, torch.int32)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        pass





































































































































































































































































