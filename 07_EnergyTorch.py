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


"""
based on
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
"""


pl.seed_everything(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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


# mnist_images = train_set.data.numpy()  # this returns the original data not transformed
# mnist_labels = train_set.targets.numpy()


# t = torch.stack([im for im, _ in train_set])


# CNN MODEL


class CNNModel(nn.Module):

    class Swish(nn.Module):

        def forward(self, x):
            return x * torch.sigmoid(x)

    def __init__(self, hidden_features=2**5, in_dims=1, out_dims=1, **kwargs):
        super().__init__()
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_dims, c_hid1, kernel_size=5, stride=2, padding=4),  # [16x16]
            self.Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # [8x8]
            self.Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            self.Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            self.Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),
            self.Swish(),
            nn.Linear(c_hid3, out_dims)
        )

    def forward(self, x):
        # return self.cnn_layers(x)
        return self.cnn_layers(x).squeeze(dim=-1)


# model = CNNModel()


# SAMPLING BUFFER


class Sampler:

    """
    We store the samples of the last couple of batches in a buffer,
    and re-use those as the starting point of the MCMC algorithm for the next batches.
    This reduces the sampling cost because the model requires a significantly
    lower number of steps to converge to reasonable samples.
    However, to not solely rely on previous samples and allow novel samples as well,
    we re-initialize 5% of our samples from scratch (random noise between -1 and 1)
    """

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [
            (torch.rand((1, ) + img_shape) * 2 - 1)
            for _ in range(self.sample_size)
        ]

    def sample_new_examples(self, steps=60, step_size=10):

        """
        :param steps: specifically set for MNIST
        :param step_size: specifically set for MNIST
        :return: new batch of fake images
        """

        # Generate 5% of batch from scratch
        n_new = np.random.binomial(self.sample_size, p=0.05)
        fake_images = torch.rand(size=(n_new, ) + self.img_shape) * 2 - 1

        # Uniformly choose (self.sample_size - n_new) elements from self.examples
        samples = random.choices(self.examples, k=self.sample_size - n_new),
        # Stacks the samples
        old_images = torch.cat(tensors=samples, dim=0)
        inp_images = (torch.cat(tensors=[fake_images, old_images], dim=0)
                      .deatch().to(device))

        # Perform MCMC sampling
        inp_images = Sampler.generate_samples(
            self.model, inp_images, steps, step_size, return_img_per_step=False
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(
            inp_images.to(torch.device("cpu")).chunck(self.sample_size, dim=0)
        ) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_images

    # SGLD
    @staticmethod
    def generate_samples(model, inp_images,
                         steps=6, step_size=10, return_img_per_step=False):

        is_training = model.training

        # set the model to evaluation as opposed to training
        model.eval()

        # set model params to requires_grad=False
        # we are only interested in gradients of input
        for p in model.parameters():
            p.requires_grad = False
        inp_images.requires_grad = True

        # enable gradient calculation if not the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # more efficient than creating a new tensor every iteration
        noise = torch.randn(size=inp_images.shape, device=inp_images.device)

        # list for storing generations at each step (for later)
        images_per_step = []

        for _ in range(steps):
            # 1) Add noise to the input
            noise.normal_(mean=0, std=0.005)  # operation in place (final _)
            inp_images.data.add(noise.data)
            inp_images.data.clamp_(min=-1., max=1.)
            # 2) Calculate gradients for the input
            out_images = - model(inp_images)
            out_images.sum().backward()
            inp_images.grad.data.clamp_(min=-0.03, max=0.03)  # stabilize gradient
            # 3) Apply gradients to current samples
            inp_images.data.add_(step_size * - inp_images.grad.data)
            inp_images.grad.detach_()
            inp_images.grad.zero_()
            inp_images.data.clamp_(min=-1., max=1.)
            # 4) add inp_images to list if needed
            if return_img_per_step:
                images_per_step.append(inp_images.clone().detach())

        # reactivate gradients of params for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # reset gradient calculation to setting before this func
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(images_per_step, dim=0)
        return inp_images


# TRAINING ALGORITHM











































































































