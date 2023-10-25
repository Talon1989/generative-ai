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


def tokenize_and_prep(text_data: list, max_seq_length=128):
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
        torch.tensor(
            [vocab[token] for token in tokenizer(item)[:max_seq_length]],
            dtype=torch.long)
        for item in text_data
    ]
    def prep_inputs(d):
        return d[:-1], d[1:]
    data = pad_sequence(data, batch_first=True)
    prepped_data = list(map(prep_inputs, data))
    dataloader = DataLoader(prepped_data, batch_size=32, shuffle=True)
    return dataloader, list(vocab.stoi.keys())


text_dataloader, vocabulary = tokenize_and_prep(text_data)
a_, b_ = next(iter(text_dataloader))


# DICTIONARY_SIZE = 10_000
EMBEDDED_VECTOR_DIM = 64
SAVE_PATH = '/home/fabio/PycharmProjects/generative-ai/data/models/lstm.pth'


class CustomLSTM(nn.Module):

    """
    for each input token, the model outputs a probability distribution
    over all possible next tokens in vocabulary
    """

    def __init__(self):
        super().__init__()
        # tokens are passed through an embedding layer to convert them into dense vectors
        self.embedding = nn.Embedding(len(vocabulary), EMBEDDED_VECTOR_DIM)
        self.lstm = nn.LSTM(
            input_size=EMBEDDED_VECTOR_DIM,
            hidden_size=128,
            num_layers=2
        )
        self.outputs = nn.Linear(128, len(vocabulary))
        self.a_softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1/1_000)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # x = self.a_softmax(self.outputs(x))
        x = self.outputs(x)  # softmax is not necessary for usage with nn.CrossEntropyLoss()
        return x

    def fit(self, n_epochs, dataloader, graph=False, save_model=False):
        avg_losses = []
        for ep in range(1, n_epochs + 1):
            cumulative_losses = []
            for a, b in dataloader:
                self.optimizer.zero_grad()
                self.train()
                b = b.view(-1)  # shape=[batch * seq_len]
                predictions = self(a)  # shape=[batch, seq_len, n_dict_tokens]
                predictions = predictions.view(-1, predictions.shape[-1])  # shape=[batch * seq_len, n_dict_tokens]
                loss = self.criterion(predictions, b)
                loss.backward()
                self.optimizer.step()
                print(loss)
                cumulative_losses.append(loss.detach().numpy())
            avg_losses.append(np.sum(cumulative_losses) / len(cumulative_losses))
            print('Ep %d | avg losses: %.4f' % (ep, avg_losses[-1]))
            if ep % 10 and graph:
                pass
            if ep % 2 and save_model:
                print('\nSaving model.')
                torch.save(self.state_dict(), PATH)
                print('Model saved.\n')


lstm = CustomLSTM()
# outputs = lstm(a_)
lstm.fit(n_epochs=10, dataloader=text_dataloader)


# preds = torch.argmax(output, dim=-1)


# criterion = nn.CrossEntropyLoss()
#
#
# preds = outputs.view(-1, outputs.shape[-1])
# targets = b_.view(-1)























































































































































































































































