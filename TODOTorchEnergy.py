import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utilities import Swish, display_image_torch


'''
based on
https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial8/Deep_Energy_Models.ipynb
'''


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


train_dataset = datasets.MNIST(
    root='data/', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(
    root='data/', train=False, transform=transform, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


images, labels = next(iter(train_dataloader))


class OldCnn(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_channels, 16, kernel_size=(5, 5), stride=2, padding=4)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.dense_1 = nn.Linear(64 * 4, 64)
        self.dense_2 = nn.Linear(64, n_channels)
        self.swish = Swish()
        self.flatten = nn.Flatten()

    def forward(self, x):
        print(x.shape)
        x = self.swish(self.conv_1(x))
        print(x.shape)
        x = self.swish(self.conv_2(x))
        print(x.shape)
        x = self.swish(self.conv_3(x))
        print(x.shape)
        x = self.swish(self.conv_4(x))
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.swish(self.dense_1(x))
        print(x.shape)
        x = self.dense_2(x)
        print(x.shape)
        return x


class Cnn(nn.Module):
    def __init__(self, n_channels=1, hidden_features=32, output_dim=1):
        super().__init__()
        c_hid_1 = hidden_features // 2
        c_hid_2 = hidden_features
        c_hid_3 = hidden_features * 2
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(n_channels, c_hid_1, kernel_size=(5, 5), stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid_1, c_hid_2, kernel_size=(3, 3), stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid_2, c_hid_3, kernel_size=(3, 3), stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid_3, c_hid_3, kernel_size=(3, 3), stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid_3*4, c_hid_3),
            Swish(),
            nn.Linear(c_hid_3, output_dim)
        )

    def forward(self, x, squeeze=True):
        x = self.cnn_layers(x)
        if squeeze:
            x = torch.squeeze(x, dim=-1)
        return x


cnn = Cnn()
output = cnn(images[0:5])
# display_image_torch(images[0])





