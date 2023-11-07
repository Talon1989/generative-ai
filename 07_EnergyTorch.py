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
import time


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
        samples = random.choices(self.examples, k=self.sample_size - n_new)
        # Stacks the samples
        old_images = torch.cat(tensors=samples, dim=0)
        inp_images = (torch.cat(tensors=[fake_images, old_images], dim=0)
                      .detach().to(device))

        # Perform MCMC sampling
        inp_images = Sampler.generate_samples(
            self.model, inp_images, steps, step_size, return_img_per_step=False
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(
            inp_images.to(torch.device("cpu")).chunk(self.sample_size, dim=0)
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
            # CONSIDER REMOVING NEGATIVE AND USE real_out.mean() - fake_out.mean() LATER
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


class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, alpha=1/10, lr=1/10_000, beta1=0., **CNN_args):
        super().__init__()
        self.save_hyperparameters()  # automatically sets arguments in __init__
        self.cnn = CNNModel(**CNN_args)
        self.sampler = Sampler(self.cnn, img_shape, batch_size)

    def forward(self, x):
        return self.cnn(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 999/1_000)
        )
        # every (step_size) epochs lr of (optimizer) is * (gamma) to be decreased
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1, gamma=97/100
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        # add some noise to og images to prevent model to focus only on 'clean' inputs
        real_images, _ = batch
        small_noise = torch.randn_like(real_images) * 0.005
        real_images.add_(small_noise).clamp_(min=-1., max=1.)  # func_ is in place

        fake_images = self.sampler.sample_new_examples(steps=60, step_size=10)

        # predict energy scores
        inp_images = torch.cat([real_images, fake_images], dim=0)
        real_out, fake_out = self.cnn(inp_images).chunk(2, dim=0)

        # calculate losses
        regularized_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        contrastive_divergence_loss = fake_out.mean() - real_out.mean()
        loss = regularized_loss + contrastive_divergence_loss

        self.log('loss', loss)
        self.log('loss_regularization', regularized_loss)
        self.log('loss_contrastive_divergence', contrastive_divergence_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        For validating, we calculate the contrastive divergence
        between purely random images and unseen examples
        """
        real_images, _ = batch
        fake_images = torch.randn_like(real_images) * 2 - 1
        inp_images = torch.cat([real_images, fake_images], dim=0)
        real_out, fake_out = self.cnn(inp_images).chunk(2, dim=0)
        contrastive_divergence = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', contrastive_divergence)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=8, n_steps=2**8, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size  # # images to generate
        self.vis_steps = vis_steps  # # steps withing generation to visualize
        self.n_steps = n_steps  # # steps to take during generation
        self.every_n_epochs = every_n_epochs  # save images every n epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            images_per_step = self.generate_images(pl_module)
            for i in range(images_per_step.shape[1]):
                step_size = self.n_steps // self.vis_steps
                images_to_plot = images_per_step[step_size-1::step_size, i]
                grid = torchvision.utils.make_grid(
                    images_to_plot,
                    nrow=images_to_plot.shape[0],
                    normalize=True,
                    range=(-1, 1)
                )
                trainer.logger.experiment.add_image(
                    'generation_%d' % i,grid, global_step=trainer.current_epoch)

    def generate_images(self, pl_module):
        pl_module.eval()
        start_images = torch.rand(
            (self.batch_size,) + pl_module.hparams['img_shape']
        ).to(pl_module.device)
        start_images = start_images * 2 - 1
        torch.set_grad_enabled(True)
        images_per_step = Sampler.generate_samples(
            pl_module.cnn, start_images,steps=self.n_steps,
            step_size=10, return_img_per_step=True
        )
        torch.set_grad_enabled(False)
        pl_module.train()
        return images_per_step


class SamplerCallback(pl.Callback):

    def __init__(self, n_images=32, every_n_epochs=5):
        super().__init__()
        self.n_images = n_images
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            example_images = torch.cat(
                random.choices(pl_module.sampler.examples, k=self.n_images),
                dim=0
            )
            grid = torchvision.utils.make_grid(
                example_images, nrow=4, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(
                'sampler', grid, global_step=trainer.current_epoch)


class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=2**10):
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            random_images = torch.rand(
                (self.batch_size, ) + pl_module.hparams['img_shape']
            ).to(pl_module.device)
            random_images = random_images * 2 - 1
            random_out = pl_module.cnn(random_images).mean()
            pl_module.train()
        trainer.logger.experiment.add_scalar(
            'random_out', random_out, global_step=trainer.current_epoch)


PATH = '/home/talon/PycharmProjects/generative-ai/data/torch-energy/MNIST'


def train_model(**kwargs):
    trainer = pl.Trainer(
        accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
        devices=1,
        gradient_clip_val=0.1,
        # default_root_dir=PATH,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode='min', monitor='val_contrastive_divergence'),
            pl.callbacks.LearningRateMonitor('epoch'),
            GenerateCallback(every_n_epochs=5),
            SamplerCallback(every_n_epochs=5),
            OutlierCallback()
        ]
    )
    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )
    model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model


train_model(img_shape=(1, 28, 28), batch_size=train_loader.batch_size)















































































