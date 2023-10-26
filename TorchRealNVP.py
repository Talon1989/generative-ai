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
from sklearn import datasets as sklearn_datasets


WEIGHT_DECAY = 1/100


data = sklearn_datasets.make_moons(n_samples=2_000, noise=1/20)[0].astype('float32')
norm = nn.BatchNorm1d(num_features=2)
normalized_data = norm(torch.tensor(data))
# plt.scatter(data[:, 0], data[:, 1], c='r')
# plt.scatter(normalized_data[:, 0].detach().numpy(), normalized_data[:, 1].detach().numpy(), c='b')
# plt.show()


class Coupling(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.s_layer_1 = nn.Linear(input_dim, 256)
        self.s_layer_2 = nn.Linear(256, 256)
        self.s_layer_3 = nn.Linear(256, 256)
        self.s_layer_4 = nn.Linear(256, 256)
        self.s_layer_5 = nn.Linear(256, input_dim)
        self.t_layer_1 = nn.Linear(input_dim, 256)
        self.t_layer_2 = nn.Linear(256, 256)
        self.t_layer_3 = nn.Linear(256, 256)
        self.t_layer_4 = nn.Linear(256, 256)
        self.t_layer_5 = nn.Linear(256, input_dim)

    def forward(self, x):
        s = self.s_layer_1(x)
        s = self.s_layer_2(s)
        s = self.s_layer_3(s)
        s = self.s_layer_4(s)
        s = self.s_layer_5(s)
        t = self.t_layer_1(x)
        t = self.t_layer_2(t)
        t = self.t_layer_2(t)
        t = self.t_layer_3(t)
        t = self.t_layer_4(t)
        t = self.t_layer_5(t)
        return s, t


coupling = Coupling()























































































































































































































































