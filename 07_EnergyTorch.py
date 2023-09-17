import os
import json
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
import torchvision
import urllib.request
from urllib.error import HTTPError

pl.seed_everything(42)

# DATASET

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_set = torchvision.datasets.MNIST(
    root='data/', train=True, transform=transform, download=True
)
test_set = torchvision.datasets.MNIST(
    root='data/', train=False, transform=transform, download=True
)
train_loader = data.DataLoader(train_set, batch_size=2 ** 7, shuffle=True,
                               drop_last=True, num_workers=4, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=2 ** 8, shuffle=False,
                              drop_last=False, num_workers=4, pin_memory=False)
