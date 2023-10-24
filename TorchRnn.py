import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


PATH = '/home/fabio/datasets/epirecipes/full_format_recipes.json'


# import json
# import re
# with open(PATH) as json_data:
#     recipe_data = json.load(json_data)
# filtered_data = [
#     "Recipe for " + x["title"] + "| " + " ".join(x["directions"])
#     for x in recipe_data
#     if "title" in x
#     and x["title"] is not None
#     and "directions" in x
#     and x["directions"] is not None
# ]


recipe_data = pd.read_json(PATH)
# recipe_data = recipe_data.dropna()
filtered_data = recipe_data[['title', 'directions']]
if filtered_data.isna().any().any():  # first any if for all rows of each columns, second is for all columns
    filtered_data = filtered_data.dropna()


# def pad_punctuation(s):
#     import re
#     import string
#     s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
#     s = re.sub(" +", " ", s)
#     return s


def pad_punctuation(s):
    import re
    import string
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


# filtered_data['directions'] = filtered_data['directions'].apply(pad_punctuation)
filtered_data = filtered_data.astype(str).map(pad_punctuation)
text_data = filtered_data['directions'].tolist()


# tokenizer = get_tokenizer('basic_english')


def tokenize_and_prep_old(text_data: list):
    from collections import Counter
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in text_data:
        counter.update(tokenizer(line))
    vocab = Vocab(
        counter=counter,
        max_size=10_000,
        specials=['<unk>', '<pad>', '<bos>', '<eos>']
    )
    # data = [
    #     torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
    #     for item in text_data
    # ]
    data = [
        torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
        for item in text_data if item.strip()
    ]
    # print(data)
    # data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    # print(data)
    # print(data.shape)
    def prep_inputs(d):
        return d[:-1], d[1:]
    prepped_data = list(map(prep_inputs, data))
    dataloader = DataLoader(prepped_data, batch_size=32, shuffle=True)
    # data = DataLoader(data, batch_size=32, shuffle=True)
    # return map(prep_inputs, data), list(vocab.stoi.keys())
    return dataloader, list(vocab.stoi.keys())


def tokenize_and_prep(text_data: list):
    from torch.nn.utils.rnn import pad_sequence
    from collections import Counter
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in text_data:
        counter.update(tokenizer(line))
    vocab = Vocab(
        counter=counter,
        max_size=10_000,
        specials=['<unk>', '<pad>', '<bos>', '<eos>']
    )
    data = [
        torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
        for item in text_data
    ]
    def prep_inputs(d):
        return d[:-1], d[1:]
    data = pad_sequence(data, batch_first=True)
    prepped_data = list(map(prep_inputs, data))
    dataloader = DataLoader(prepped_data, batch_size=32, shuffle=True)
    return dataloader, list(vocab.stoi.keys())


train_dataloader, vocabulary = tokenize_and_prep(text_data)







































































































































































































































































