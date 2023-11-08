import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.utils
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utilities import Swish, display_image_torch
import lightning as pl


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

    # SGLD
    # POSSIBLE ISSUES WITH .BACKWARD AND .GRAD METHODS
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
            noise.normal_(0, 5/1_000)  # omega
            inp_images.data.add_(noise.data)
            inp_images.data.clamp_(min=-1., max=1.)
            # calculate gradients for current input
            energy_values = - model(inp_images)  # negative value of energy function output for inp_images
            # gradients of the scalar tensor resulting from the sum() method applied
            # to the energy_values tensor with respect to the inp_images tensor
            energy_values.sum().backward()
            # Since the backward() method computes the gradients of the tensor
            # with respect to the inp_images tensor,
            # the gradients can be accessed using inp_images.grad.data
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


class EnergyModel(pl.LightningModule):
    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1/10_000, beta=0., **CNN_args):
        super().__init__()
        self.save_hyperparameters()
        self.energy_model = Cnn(**CNN_args)
        self.sampler = Sampler(self.energy_model, img_shape, batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.energy_model(x)
        return z

    # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
    # Hence, we set it to 0 by default.
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=97/100)  # exp decay
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        small_noise = torch.randn_like(real_images) * 5/1_000  # small gaussian noise
        # add small noise to og images to prevent model from focus only on 'clean' inputs
        real_images.add(small_noise).clamp_(min=-1., max=1.)
        fake_images = self.sampler.sample_new_examples()
        inp_images = torch.cat([real_images, fake_images], dim=0)
        real_energy, fake_energy = self.energy_model(inp_images).chunk(2, dim=0)
        reg_loss = self.hparams.alpha * (real_energy ** 2 + fake_energy ** 2).mean()
        cd_loss = fake_energy.mean() - real_energy.mean()
        # cd_loss = real_energy.mean() - fake_energy.mean()
        loss = reg_loss + cd_loss
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cd_loss)
        self.log('metrics_avg_real', real_energy.mean())
        self.log('metrics_avg_fake', fake_energy.mean())
        return loss

    # For validating, we calculate the contrastive divergence between purely random images and unseen examples
    # Note that the validation/test step of energy-based models depends on what we are interested in the model
    def validation_step(self, batch, batch_idx):
        real_images, _ = batch
        fake_images = torch.rand(real_images) * 2 - 1
        inp_images = torch.cat([real_images, fake_images], dim=0)
        real_energy, fake_energy = self.energy_model(inp_images).chunk(2, dim=0)
        cd_loss = fake_energy.mean() - real_energy.mean()
        # cd_loss = real_energy.mean() - fake_energy.mean()
        self.log('val_contrastive_divergence', cd_loss)
        self.log('val_fake_out', fake_energy.mean())
        self.log('val_real_out', real_energy.mean())


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=8, vis_steps=8, n_steps=2**8, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size
        self.vis_steps = vis_steps
        self.n_steps = n_steps
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer:pl.LightningModule, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            images = self.generate_images(pl_module)
            for i in range(images[1]):
                step_size = self.n_steps // self.vis_steps
                images_to_plot = images[step_size-1::step_size, 1]
                grid = torchvision.utils.make_grid(
                    images_to_plot, nrow=images_to_plot.shape[0], normalize=True, range=(-1, 1))
                trainer.logger.experiment.add_image(
                    'Generation %d' % i, grid, global_step=trainer.current_epoch)

    def generate_images(self, pl_module:pl.LightningModule):
        pl_module.eval()
        start_images = torch.rand((self.batch_size,) + pl_module.hparams['img_shape']).to(pl_module.device)
        start_images = start_images * 2 - 1.
        torch.set_grad_enabled(True)  # tracking gradient for sampling
        images = Sampler.generate_sample(
            pl_module.energy_model, start_images, n_steps=self.n_steps, step_size=10, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return images


# adds randomly picked subset of images
class SamplerCallback(pl.Callback):
    def __init__(self, n_images=32, every_n_epochs=5):
        super().__init__()
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            images = torch.cat(random.choices(pl_module.sampler.examples, k=self.n_images), dim=0)
            grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image('sampler', grid, global_step=trainer.current_epoch)


# record negative energy assigned to random noise
class OutlierCallback(pl.Callback):
    def __init__(self, batch_size=2**10):
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_images = torch.rand((self.batch_size,) + pl_module.hparams['img_shape']).to(pl_module.device)
            rand_images = rand_images * 2 - 1.
            rand_energy = pl_module.energy_model(rand_images).mean()
            pl_module.train()
        trainer.logger.experiment.add_scalar('rand_energy', rand_energy, global_step=trainer.current_epoch)


def train_model(**kwargs):
    pass










