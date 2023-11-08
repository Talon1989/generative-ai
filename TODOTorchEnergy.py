import random
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# Store the samples of the last couple of batches in a buffer,
# and re-use those as the starting point of the MCMC algorithm for the next batches.
# This reduces the sampling cost because the model requires a significantly lower
# number of steps to converge to reasonable samples. However, to not solely rely on
# previous samples and allow novel samples as well, we re-initialize 5% of our samples
# from scratch (random noise between -1 and 1)
class Sampler:
    def __init__(self, model, image_shape, sample_size, max_len=2**13):
        self.model = model  # nn used for modeling e_theta
        self.image_shape = image_shape
        self.sample_size = sample_size
        self.max_len = max_len  # max number of datapoints to keep in the buffer
        self.examples = [
            (torch.rand((1, ) + image_shape) * 2 - 1)
            for _ in range(self.sample_size)
        ]

    # Sample a new batch of fake images with probability p
    def sample_new_examples(self, steps:int=60, step_size:int=10, p=5/100):
        n_new = np.random.binomial(n=self.sample_size, p=p)
        rand_images = torch.rand((n_new,) + self.image_shape) * 2 - 1
        old_images = torch.cat(
            random.choices(self.examples, k=self.sample_size - n_new),
            dim=0)
        inp_images = torch.cat([rand_images, old_images], dim=0).detach().to(device)
        imp_images = Sampler.generate_sample(self.model, inp_images, steps, step_size)
        self.examples = list(
            inp_images.to(torch.device('cpu')
                          ).chunk(chunks=self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return imp_images

    @staticmethod
    def generate_sample(model:nn.Module, inp_images:torch.Tensor,
                        n_steps=60, step_size=10, return_img_per_step=False):
        # before mcmc: since we are only interested in the gradient of the input,
        # set the model parameters to required_grad = False
        is_training = model.trainning
        for p in model.parameters():
            p.requires_grad = False
        inp_images.requires_grad = True
        old_grads_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        noise = torch.randn(inp_images.shape, device=inp_images.device)
        images_per_step = []
        for _ in range(n_steps):
            # add noise to the input
            noise.normal_(0, 5/1_000)
            inp_images.data.add_(noise.data)
            inp_images.data.clamp_(min=-1., max=1.)
            # calculate gradients for current input
            out_images = - model(inp_images)
            out_images.sum().backward()
            inp_images.grad.data.clamp_(min=-0.03, max=0.03)  # stabilize and prevent high gradient
            # apply gradients to current samples
            inp_images.data.add_(-step_size * inp_images.data)
            inp_images.grad.detach_()
            inp_images.grad.zero_()
            inp_images.data.clamp_(min=-1., max=1.)
            #
            if return_img_per_step:
                images_per_step.append(inp_images.clone().detach())
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(old_grads_enabled)
        if return_img_per_step:
            return torch.stack(images_per_step, dim=0)
        else:
            return inp_images

