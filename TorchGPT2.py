import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

"""
based on
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""


class PositionalEncoding(nn.Module):

    """
    PositionalEncoding module injects some information about the relative
    or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings
    so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 1/10, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(dim=1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.) / d_model))
        pe = torch.zeros([max_len, 1, d_model])
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# class Transformer(nn.Module):
#
#     def __init__(self, n_token: int, d_model: int, n_head:int,
#                  d_hid: int, n_layers:int, dropout: float = 1/2):
#         super().__init__()
#
#
# train_iter = WikiText2(split='train')




































































































































































































