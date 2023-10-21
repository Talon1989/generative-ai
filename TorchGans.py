import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


LEGO_PATH = '/home/fabio/datasets/lego-brick-images/'


lego_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
lego_dataset = datasets.ImageFolder(
    root=LEGO_PATH,
    transform=lego_transform
)
lego_dataloader = DataLoader(lego_dataset, batch_size=64, shuffle=True)


mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
mnist_dataset = datasets.MNIST(
    root='../', train=True, download=True, transform=mnist_transform
)
mnist_dataloader = DataLoader(dataset=mnist_dataset, shuffle=True, batch_size=64)


# images, labels = next(iter(lego_dataloader))
# image = images[5].squeeze()
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()


images, labels = next(iter(mnist_dataloader))
image = images[5].squeeze()
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()


class Discriminator(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_channels, 64, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_4 = nn.Conv2d(256, 1, kernel_size=(4, 4), stride=1, padding=0)
        self.batch_norm_2 = nn.BatchNorm2d(128, momentum=9/10)
        self.batch_norm_3 = nn.BatchNorm2d(256, momentum=9/10)
        self.swish = Swish()
        self.leaky_relu = nn.LeakyReLU(negative_slope=1/5)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=3/10)
        self.flatten = nn.Flatten()

    def forward(self, x:torch.Tensor):
        x = self.swish(self.conv_1(x))
        x = self.dropout(x)
        x = self.swish(self.conv_2(x))
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = self.swish(self.conv_3(x))
        x = self.batch_norm_3(x)
        x = self.dropout(x)
        x = self.sigmoid(self.conv_4(x))
        x = self.flatten(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class GenerativeAdversarialNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


discriminator = Discriminator(n_channels=1)








































































































































































































































































































































































































































































































