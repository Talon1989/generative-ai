import numpy as np
import copy
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities import Swish, display_image_torch


'''
trying to make diffusion model work in pytorch
'''


PATH = '/home/fabio/.kaggle/datasets/pytorch-challange-flower-dataset/dataset/'
EMA = 999 / 1_000
EMBEDDED_DIM = 32
IMAGE_SIZE = 64
BATCH_SIZE = 128


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(
    root=PATH, transform=transform)
train_size = int(8/10 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, (len(dataset) - train_size)])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


images, labels = next(iter(train_dataloader))
# image = images[0].permute((1, 2, 0))
# plt.imshow(image)
# plt.axis('off')
# plt.show()
# plt.clf()


def cosine_diffusion_schedule(diffusion_times):
    if type(diffusion_times) is not torch.Tensor:
        diffusion_times = torch.Tensor(diffusion_times)
    noise_rates = torch.sin(diffusion_times * (torch.pi / 2))
    signal_rates = torch.cos(diffusion_times * (torch.pi / 2))
    return noise_rates, signal_rates


def sinusoidal_embedding(x):
    start = np.log(1.)
    end = np.log(1000.)
    n_samples = EMBEDDED_DIM // 2
    frequencies = torch.exp(torch.linspace(start, end, n_samples))
    frequencies = frequencies.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    angular_speeds = frequencies * (2. * torch.pi)
    embeddings = torch.cat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)],
        dim=1
    )
    return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, input_shape: int, width: int):
        super().__init__()
        self.width = width
        self.utility_conv = nn.Conv2d(input_shape, self.width, kernel_size=1)
        self.conv_1 = nn.Conv2d(input_shape, self.width, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1)
        self.normalizer = nn.BatchNorm2d(input_shape, affine=False)
        self.swish = Swish()

    def forward(self, x):
        n_channels = x.shape[1]
        if n_channels == self.width:
            residual = x
        else:  # increase number of channels
            residual = self.utility_conv(x)
        x = self.normalizer(x)
        x = self.swish(self.conv_1(x))
        x = self.conv_2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: np.array, width: int):
        super().__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_c, width) for in_c in in_channels
        ])
        self.down_sample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: tuple):
        x, skips = x
        for block in self.residual_blocks:
            x = block(x)
            skips.append(x)
        x = self.down_sample(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_channels: np.array, width: int):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_c, width) for in_c in in_channels
        ])

    def forward(self, x: tuple):
        x, skips = x
        x = self.up_sample(x)
        for block in self.residual_blocks:
            x = torch.cat([x, skips.pop()], dim=1)  # channel dimension
            x = block(x)
        return x, skips





























































































































































































































