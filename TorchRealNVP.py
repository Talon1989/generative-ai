import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import datasets as sklearn_datasets


WEIGHT_DECAY = 1/100


data_ = sklearn_datasets.make_moons(n_samples=2_000, noise=1/20)[0].astype('float32')
norm = nn.BatchNorm1d(num_features=2)
normalized_data = norm(torch.tensor(data_))
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
        t = self.t_layer_3(t)
        t = self.t_layer_4(t)
        t = self.t_layer_5(t)
        return s, t


coupling = Coupling()


class RealNVP(nn.Module):
    def __init__(self, input_dim, n_coupling_layers):
        super().__init__()
        self.standard_gaussian = MultivariateNormal(
            loc=torch.tensor([0. for _ in range(input_dim)]),
            covariance_matrix=torch.diag(torch.tensor([1. for _ in range(input_dim)]))
        )
        # self.masks = torch.tensor(
        #     [[0., 1.], [1., 0.]] * (n_coupling_layers // 2)
        # )
        mask_arrays = [[0. for _ in range(input_dim)] for _ in range(input_dim)]
        for i in range(input_dim):
            mask_arrays[(input_dim-1) - i][i] = 1.
        self.masks = torch.tensor(
            mask_arrays * (n_coupling_layers // 2)
        )
        # self.coupling_layers = [Coupling(input_dim) for _ in range(n_coupling_layers)]
        self.coupling_layers = nn.ModuleList()
        for _ in range(n_coupling_layers):
            self.coupling_layers.append(Coupling(input_dim))

    def forward(self, x, training=False):

        log_det_inv = 0
        direction = -1 if training else 1

        for i in range(len(self.coupling_layers))[::direction]:

            reverse_mask = 1 - self.masks[i]

            x_masked = x * self.masks[i]
            s, t = self.coupling_layers[i](x_masked)
            s = s * reverse_mask
            t = t * reverse_mask

            gate = (direction - 1) / 2  # gate is close (gate=0) if direction=1, i.e., Training=False

            output = (
                reverse_mask *
                (x * torch.exp(s * direction)) + (t * direction * torch.exp(s * gate))
            ) + x_masked
            log_det_inv = log_det_inv + (torch.sum(s, dim=-1) * gate)

            return output, log_det_inv

    def log_loss(self, x):
        output, log_det = self(x, training=True)
        # .log_prob takes ln of the value f(x) where f is the pdf (the actual y value in the distribution)
        log_likelihood = self.standard_gaussian.log_prob(output) + log_det
        return torch.mean(-log_likelihood)


realnvp = RealNVP(input_dim=2, n_coupling_layers=4)
optimizer = torch.optim.Adam(params=realnvp.parameters(), lr=1/100_000, weight_decay=WEIGHT_DECAY)


def fit(model, optim, data, n_epochs=500):
    losses = []
    for ep in range(1, n_epochs+1):
        optim.zero_grad()
        model.train()
        loss = model.log_loss(data)
        loss.backward(retain_graph=True)  # need this for iterating through nn.ModuleList()
        optim.step()
        losses.append(loss.detach().numpy())
        print('Episode %d | loss: %.4f' % (ep, losses[-1]))


fit(realnvp, optimizer, normalized_data)









































































































































































































































