import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


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












































































































































































































































































































































































































































































































