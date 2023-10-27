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
image = images[0].permute((1, 2, 0))
plt.imshow(image)
plt.axis('off')
plt.show()
plt.clf()

#




























































































































































































































