import os

import numpy as np
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import time

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from tempfile import TemporaryDirectory

"""
based on
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


EM_SIZE = 200  # embedding dimension
D_HID = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
N_LAYERS = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
N_HEAD = 2  # number of heads in ``nn.MultiheadAttention``
DROPOUT = 0.2  # dropout probability
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
GRAD_CLIP_VALUE = 1/2
LR = 5.  # follows StepLR schedule
N_EPOCHS = 3


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module injects some information about the relative
    or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings
    so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 1 / 10, max_len: int = 5_000):
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


# pos_enc = PositionalEncoding(d_model=EM_SIZE)


class Transformer(nn.Module):
    def __init__(self, n_token: int, d_model: int, n_head: int,
                 d_hid: int, n_layers: int, dropout: float = 1 / 2):
        super().__init__()

        def init_weights() -> None:
            init_range = 1 / 10
            with torch.no_grad():
                # initializes weights sampling from U([-init_range, +init_range)
                self.embedding.weight.data.uniform_(-init_range, +init_range)
                self.linear.bias.data.zero_()
                # initializes weights sampling from U([-init_range, +init_range)
                self.linear.weight.data.uniform_(-init_range, +init_range)

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=n_layers)
        self.embedding = nn.Embedding(
            num_embeddings=n_token, embedding_dim=d_model)
        self.d_model = d_model
        self.linear = nn.Linear(in_features=d_model, out_features=n_token)
        init_weights()

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        :param src: Tensor, shape: [batch_size, seq_len]
        :param src_mask: Tensor, shape: [seq_len, seq_len]
        :return: Tensor, shape: [batch_size, seq_len, n_token]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


# optimus_prime = Transformer(10_000, EM_SIZE, 4, 16, 16)


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    map(tokenizer, train_iter), specials=['<unk>'])


N_TOKENS = len(vocab)


# print(vocab(tokenizer('this is a potato')))


# converts raw text into flat tensor
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)


# REWRITE USING DATASETS / DATALOADER
def batchify(data: Tensor, bsz: int) -> Tensor:
    """
    Divides the data into ``bsz`` separate sequences,
    removing extra elements that wouldn't cleanly fit.
    """
    seq_len = data.shape[0] // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


train_data_batched = batchify(train_data, BATCH_SIZE)
val_data_batched = batchify(val_data, EVAL_BATCH_SIZE)
test_data_batched = batchify(test_data, EVAL_BATCH_SIZE)


bptt = 35  # back-propagation through time


def get_batch(source: Tensor, i: int) -> tuple[Tensor, Tensor]:
    """
    :param source: Tensor, shape: [batch_size, full_seq_len]
    :param i: int
    :return: (data, target) where data shape: [batch_size, seq_len]
                                target shape: [batch_size * seq_len]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    # target = torch.reshape(
    #     source[i+1: i+1+seq_len], -1
    # )
    return data, target


model_ = Transformer(N_TOKENS, EM_SIZE, N_HEAD, D_HID, N_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=1, gamma=0.95)


def train(model: nn.Module, epoch: int) -> None:

    model.train()  # turn on train mode

    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    n_batches = len(train_data_batched) // bptt

    for batch, i in enumerate(range(0, train_data_batched.shape[0] - 1, bptt)):

        data, targets = get_batch(train_data_batched, i)
        output = model(data)
        output_flat = output.view(-1, N_TOKENS)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                'Epoch %d | %d/%d batches | lr: %.3f | ms/batch: %.3f | loss: %.3f| ppl: %.3f' %
                (epoch, batch, n_batches, lr, ms_per_batch, cur_loss, ppl)
            )
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data_b: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data_b.shape[0] - 1, bptt):
            data, targets = get_batch(eval_data_b, i)
            seq_len = data.shape[0]
            output = model(data)
            output_flat = output.view(-1, N_TOKENS)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data_b) - 1)


best_val_loss = float('inf')
save_path = os.getcwd() + '/data/models/pytorch_GPT.pth'


# with TemporaryDirectory() as temp_dir:
#     best_model_params_path = os.path.join(temp_dir, 'best_model_params.pth')
#     for ep in range(1, N_EPOCHS+1):
#         ep_start_time = time.time()
#         train(model_, ep)
#         val_loss = evaluate(model_, val_data_batched)
#         val_ppl = math.exp(val_loss)
#         elapsed_time = time.time() - ep_start_time
#         print('-' * 89)
#         print('End of epoch %d | time: %.3fs | val loss: %.3f | val ppl: %.3f' %
#               (ep, elapsed_time, val_ppl, val_ppl))
#         print('-' * 89)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model_.state_dict(), best_model_params_path)
#         scheduler.step()
#     model_.load_state_dict(torch.load(best_model_params_path))


for ep in range(1, N_EPOCHS+1):
    ep_start_time = time.time()
    train(model_, ep)
    val_loss = evaluate(model_, val_data_batched)
    val_ppl = math.exp(val_loss)
    elapsed_time = time.time() - ep_start_time
    print('-' * 89)
    print('End of epoch %d | time: %.3fs | val loss: %.3f | val ppl: %.3f' %
          (ep, elapsed_time, val_ppl, val_ppl))
    print('-' * 89)
    if val_loss < best_val_loss:
        print('Saving better model')
        best_val_loss = val_loss
        torch.save(model_.state_dict(), save_path)
    scheduler.step()
model_.load_state_dict(torch.load(save_path))
