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
BATCH_SIZE = 64


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


class UNET(nn.Module):
    def __init__(self, n_channel: int, noise_embedding):
        super().__init__()
        self.input_conv = nn.Conv2d(n_channel, 32, kernel_size=1)
        self.noise_embedding = noise_embedding
        self.upsample = nn.Upsample(scale_factor=64, mode='nearest')
        self.down_block_1 = DownBlock(in_channels=[64, 32], width=32)
        self.down_block_2 = DownBlock(in_channels=[32, 64], width=64)
        self.down_block_3 = DownBlock(in_channels=[64, 96], width=96)
        self.residual_1 = ResidualBlock(input_shape=96, width=128)
        self.residual_2 = ResidualBlock(input_shape=128, width=128)
        self.up_block_1 = UpBlock(in_channels=[224, 192], width=96)
        self.up_block_2 = UpBlock(in_channels=[160, 128], width=64)
        self.up_block_3 = UpBlock(in_channels=[96, 64], width=32)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.output_conv.weight.data.fill_(0.)

    def forward(self, x: tuple):
        noisy_images, noise_variance = x
        noisy_images = self.input_conv(noisy_images)
        noise_variance = self.noise_embedding(noise_variance)
        noise_variance = self.upsample(noise_variance)
        x = torch.cat([noisy_images, noise_variance], dim=1)  # concat on channels
        skips = []
        x, skips = self.down_block_1((x, skips))
        x, skips = self.down_block_2((x, skips))
        x, skips = self.down_block_3((x, skips))
        x = self.residual_1(x)
        x = self.residual_2(x)
        x, skips = self.up_block_1((x, skips))
        x, skips = self.up_block_2((x, skips))
        x, _ = self.up_block_3((x, skips))
        x = self.output_conv(x)
        return x


# WITHOUT NORMALIZER
class DiffusionModel(nn.Module):
    def __init__(self, unet_model: UNET, diff_schedule):
        super().__init__()
        self.unet = unet_model
        self.ema_unet = copy.deepcopy(self.unet)
        self.diff_schedule = diff_schedule
        self.unet_optimizer = torch.optim.AdamW(
            params=self.unet.parameters(),
            lr=1/1_000,
            weight_decay=1/10_000
        )
        self.unet_criterion = nn.MSELoss()

    def ema_soft_update(self):
        for w, ema_w in zip(self.unet.parameters(), self.ema_unet.parameters()):
            ema_w.data = EMA * ema_w.data + (1. - EMA) * w.data

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            self.unet.train()
            pred_noises = self.unet((noisy_images, noise_rates ** 2))
        else:
            self.ema_unet.eval()
            with torch.no_grad():
                pred_noises = self.ema_unet((noisy_images, noise_rates ** 2))
        pred_images = (noisy_images - (noise_rates * pred_noises)) / signal_rates
        return pred_noises, pred_images

    def train_step(self, images):
        pass

    def val_step(self, images):
        pass

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        pass

    def generate(self, n_images, diffusion_steps):
        pass


noises = torch.randn_like(images)
batch_size = images.shape[0]
diffusion_times = torch.rand(size=(batch_size, 1, 1, 1))
noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times)


unet = UNET(3, sinusoidal_embedding)
# print(
#     unet((images, noise_rates))
# )


diffusion_model = DiffusionModel(unet, cosine_diffusion_schedule)








































































































































































































