import numpy as np
import copy
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchmetrics
from custom_modules_torch import ResidualBlock, DownBlock, UpBlock
from utilities import display_images_torch, display_image_torch


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


PATH = "/home/fabio/.kaggle/datasets/pytorch-challange-flower-dataset/dataset/train/"
EMA = 999 / 1_000
NOISE_EMBEDDING_SIZE = 32
IMAGE_SIZE = 64
BATCH_SIZE = 64


# train_data = keras.utils.image_dataset_from_directory(
#     directory=PATH,
#     labels=None,
#     image_size=[64, 64],
#     batch_size=None,
#     shuffle=True,
#     seed=42,
#     interpolation="bilinear",
# )


def image_dataloader_from_directory(directory, image_size=(64, 64), batch_size=64, shuffle=True):
    transform_pipeline = torchvision.transforms.Compose([
        # Interpolation is defaulted to 'bilinear'
        torchvision.transforms.Resize(image_size),
        # This will convert to torch.float32 in the range [0.0, 1.0]
        torchvision.transforms.ToTensor(),
    ])
    # ImageFolder will automatically create a label value for each image
    dataset = torchvision.datasets.ImageFolder(
        root=directory, transform=transform_pipeline
    )
    torch.manual_seed(42)
    # dataloader = data.DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle
    # )
    # return dataloader
    # DIVING INTO TRAIN AND VALIDATION DATA
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    train_dataloader = data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader


# shape is (64, 3, 64, 64): (batch_size, channels, dim_1, dim_2)
train_data, val_data = image_dataloader_from_directory(directory=PATH)
# print(train_data.dataset.classes)
# for imgs, lbs in train_data:
#     images, labels = imgs, lbs
#     print(images.shape)
#     break
# images, labels = next(iter(train_data))
# images = images.numpy()


def linear_diffusion_schedule(diffusion_times):
    min_rate = 1 / 10_000
    max_rate = 1 / 50
    betas = min_rate + torch.tensor(diffusion_times) * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # cumulative product
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    if type(diffusion_times) is not torch.Tensor:
        diffusion_times = torch.tensor(diffusion_times)
    noise_rates = torch.sin(diffusion_times * torch.pi / 2)
    signal_rates = torch.cos(diffusion_times * torch.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = torch.tensor(1 / 50)
    max_signal_rate = torch.tensor(19 / 20)
    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


# def sinusoidal_embedding(x):
#     frequencies = torch.exp(
#         torch.linspace(
#             torch.log(torch.tensor(1.)),  # start
#             torch.log(torch.tensor(1000.)),  # end
#             NOISE_EMBEDDING_SIZE // 2
#         )
#     )
#     angular_speeds = 2.0 * torch.pi * frequencies
#     embeddings = torch.cat(
#         [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)],
#         dim=3
#     )
#     return embeddings


def sinusoidal_embedding(x):
    frequencies = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.)),  # start
            torch.log(torch.tensor(1000.)),  # end
            NOISE_EMBEDDING_SIZE // 2
        )
    ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    angular_speeds = 2.0 * torch.pi * frequencies
    embeddings = torch.cat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)],
        dim=1
    )
    return embeddings


class UNET(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.image_conv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.noise_embedding = sinusoidal_embedding
        self.noise_upsampling = nn.Upsample(scale_factor=64, mode='nearest')
        self.down_block_1 = DownBlock(in_channels=[64, 32], width=32, block_depth=2)
        self.down_block_2 = DownBlock(in_channels=[32, 64], width=64, block_depth=2)
        self.down_block_3 = DownBlock(in_channels=[64, 96], width=96, block_depth=2)
        self.residual_block_1 = ResidualBlock(in_channels=96, width=128)
        self.residual_block_2 = ResidualBlock(in_channels=128, width=128)
        self.up_block_1 = UpBlock(in_channels=[224, 192], width=96, block_depth=2)
        self.up_block_2 = UpBlock(in_channels=[160, 128], width=64, block_depth=2)
        self.up_block_3 = UpBlock(in_channels=[96, 64], width=32, block_depth=2)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        self.out_conv.weight.data.fill_(0.)  # initialize out_conv weights to zero values

    def forward(self, x):

        noisy_images, noise_variance = x

        noisy_images = self.image_conv(noisy_images)
        noise_variance = self.noise_embedding(noise_variance)
        noise_variance = self.noise_upsampling(noise_variance)

        x = torch.cat([noisy_images, noise_variance], dim=1)

        skips = []

        x, skips = self.down_block_1((x, skips))
        x, skips = self.down_block_2((x, skips))
        x, skips = self.down_block_3((x, skips))

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        x, skips = self.up_block_1((x, skips))
        x, skips = self.up_block_2((x, skips))
        x, _ = self.up_block_3((x, skips))

        x = self.out_conv(x)

        return x


PATH = "/home/talon/datasets/flower-dataset/dataset"
NORMALIZER_PATH = '/home/talon/PycharmProjects/generative-ai/data/normalizer_data'


class DiffusionModel(nn.Module):
    def __init__(self, unet_model: UNET, diff_schedule):
        super().__init__()
        self.network = unet_model
        self.ema_network = copy.deepcopy(self.network)
        # self.normalizer = None
        self.last_batch_size = 0  # util to be used with normalizer
        self.diffusion_schedule = diff_schedule
        self.noise_loss_tracker = torchmetrics.MeanMetric()
        self.optimizer = torch.optim.AdamW(
            params=self.network.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        # self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()

    # @staticmethod
    # def get_normalizer(images):
    #     rgb_mean = images.mean(dim=[0, 2, 3])  # not on 1 because it's rbg
    #     rgb_std = images.std(dim=[0, 2, 3])  # not on 1 because it's rbg
    #     normalize_transform = torchvision.transforms.Normalize(rgb_mean, rgb_std)
    #     return normalize_transform

    def update_stats(self, prev_mean, prev_std, prev_count, images):
        # Compute current batch statistics
        batch_mean = images.mean(dim=[0, 2, 3])
        batch_std = images.std(dim=[0, 2, 3])
        batch_count = images.shape[0]

        updated_mean = (prev_mean * prev_count + batch_mean * batch_count) / (prev_count + batch_count)

        batch_var = batch_std ** 2
        prev_var = prev_std ** 2
        updated_var = (prev_count * prev_var + batch_count * batch_var +
                       (prev_count * batch_count * (prev_mean - batch_mean) ** 2) / (prev_count + batch_count)) / (
                                  prev_count + batch_count)

        updated_std = torch.sqrt(updated_var)

        self.last_batch_size = batch_count

        return updated_mean, updated_std

    # def update_normalizer(self, images):
    #     if self.normalizer is None:
    #         means, stds = self.update_stats(0, 0, 0, images)
    #         self.normalizer = torchvision.transforms.Normalize(means, stds)
    #     else:
    #         prev_means, prev_stds = self.normalizer.mean, self.normalizer.std
    #         means, stds = self.update_stats(prev_means, prev_stds, self.last_batch_size, images)
    #         self.normalizer = torchvision.transforms.Normalize(means, stds)

    def update_normalizer(self, images, normalizer_path):
        if os.path.getsize(normalizer_path) == 0:  # file is empty
            means, stds = self.update_stats(0, 0, 0, images)
            self.save_means_stds(normalizer_path, means, stds)
        else:
            prev_means, prev_stds = self.get_means_stds(normalizer_path)
            means, stds = self.update_stats(prev_means, prev_stds, self.last_batch_size, images)
            self.save_means_stds(normalizer_path, means, stds)

    def reset_normalizer(self, normalizer_path):
        with open(normalizer_path, 'w') as f:
            pass

    def ema_soft_update(self):
        for weight, ema_weight in zip(self.network.parameters(), self.ema_network.parameters()):
            ema_weight.data.copy_(EMA * ema_weight.data + (1.0 - EMA) * weight.data)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            self.network.train()
            pred_noises = self.network([noisy_images, noise_rates ** 2])
        else:
            self.ema_network.eval()
            with torch.no_grad():
                pred_noises = self.ema_network([noisy_images, noise_rates ** 2])
                # pred_noises = self.network([noisy_images, noise_rates ** 2])
        pred_images = (noisy_images - (noise_rates * pred_noises)) / signal_rates
        # with torch.no_grad():
        #     print('pred_noises mean value:%.3f | std:%.3f' %
        #           (torch.mean(pred_noises).numpy(), torch.std(pred_noises).numpy()))
        return pred_noises, pred_images

    def train_step(self, images, normalizer_path):

        # self.update_normalizer(images, normalizer_path)
        # self.normalize(images, normalizer_path)

        # images, _ = data  # torch data.DataLoader automatically creates a label list
        noises = torch.randn_like(images)
        batch_size = images.shape[0]

        # sample diffusion times from uniform [0, 1]
        diffusion_times = torch.rand(size=(batch_size, 1, 1, 1))
        # use them to generate noise and signal rates
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = (signal_rates * images) + (noise_rates + noises)

        self.optimizer.zero_grad()
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True
        )
        noise_loss = self.criterion(noises, pred_noises)
        noise_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
        self.optimizer.step()

        self.ema_soft_update()

        self.noise_loss_tracker.update(noise_loss)
        print('Noise loss : %.4f\n' % self.noise_loss_tracker.compute().item())
        # self.update_normalizer(images)

    def val_step(self, images, normalizer_path):
        # self.normalize(images, normalizer_path)
        noises = torch.randn_like(images)
        batch_size = images.shape[0]
        diffusion_times = torch.rand(size=(batch_size, 1, 1, 1))
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = (signal_rates * images) + (noise_rates + noises)
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        loss = self.criterion(noises, pred_noises)
        return loss.item()

    def reverse_diffusion(self, initial_noise, diffusion_steps):

        n_images = initial_noise.shape[0]
        step_size = 1. / diffusion_steps
        current_images = initial_noise
        pred_images = None

        for step in range(diffusion_steps):

            diffusion_times = torch.ones(size=(n_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            display_image_torch(image=pred_images[0])

            next_diffusion_times = diffusion_times - step_size
            # next_diffusion_times = diffusion_times + step_size
            next_noise_rates, next_signal_rate = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rate * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def get_means_stds(self, normalizer_path):
        with open(normalizer_path, 'r') as f:
            lines = f.readlines()
            means = torch.tensor(list(map(float, lines[0].split())))
            stds = torch.tensor(list(map(float, lines[1].split())))
        return means, stds

    def save_means_stds(self, normalizer_path, means, stds):
        with open(normalizer_path, 'w') as f:
            f.write(' '.join(map(str, means.numpy()))+'\n')
            f.write(' '.join(map(str, stds.numpy()))+'\n')

    def normalize(self, images, normalizer_path):
        means, stds = self.get_means_stds(normalizer_path)
        images = (means.view(1, 3, 1, 1) - images) / stds.view(1, 3, 1, 1)
        return torch.clip(images, min=0., max=1.)

    def denormalize(self, images, normalizer_path):
        means, stds = self.get_means_stds(normalizer_path)
        images = means.view(1, 3, 1, 1) + (images * stds.view(1, 3, 1, 1))
        return torch.clip(images, min=0., max=1.)

    def generate(self, n_images, diffusion_steps, normalizer_path):
        initial_noise = torch.rand(size=(n_images, 3, 64, 64))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        # generations = self.denormalize(generated_images, normalizer_path)
        generations = torch.clip(generated_images, min=0., max=1.)
        return generations


# diffusion_times = torch.ones(BATCH_SIZE, 1, 1, 1, device=device) - 1/10 * 2
# a, b = cosine_diffusion_schedule(diffusion_times)
# images, labels = next(iter(train_data))
# output = unet([images, a])


# # torch multivariate normal outputs
# means = torch.tensor([0, 1, 2], dtype=torch.float32).unsqueeze(0).expand(5, -1)
# stds = torch.tensor([1, .9, .8], dtype=torch.float32).unsqueeze(0).expand(5, -1)
# torch.normal(means, stds)  # no need to specify size in this case


unet = UNET(3)
diffusion_model = DiffusionModel(unet, cosine_diffusion_schedule)


# training without custom normalization
def train_diffusion(model, n_epochs, model_path, save_model=False):
    total_batches = len(train_data)
    for epoch in range(n_epochs):
        # model.reset_normalizer(NORMALIZER_PATH)
        start_time = time.time()
        for batch_number, (images, _) in enumerate(train_data, start=1):
            print('Epoch: %d | Batch %d/%d:' % (epoch + 1, batch_number, total_batches))
            model.train_step(images, NORMALIZER_PATH)
        # model.save_normalizer(NORMALIZER_PATH)
        # model.normalizer = None  # resetting normalizer

        if save_model:
            torch.save(model.state_dict(), f=model_path+'.pth')
            print('Model saved in %s' % (model_path+'.pth'))
        end_time = time.time()
        print('Elapsed time for training epoch %d : %.3f minutes' % (epoch + 1, (end_time - start_time)/60))
        # RUNNING VALIDATION
        cumulative_validation_loss = 0
        for batch_number, (images, _) in enumerate(val_data, start=1):
            cumulative_validation_loss += model.val_step(images, NORMALIZER_PATH)
        print('Cumulative validation loss %.4f' % cumulative_validation_loss)


M_PATH = '/home/talon/PycharmProjects/generative-ai/data/models/U-Net-Pytorch'


# LOAD TORCH MODEL
# model_state_dict = torch.load(M_PATH+'.pth')
# loaded_model = DiffusionModel(UNET(3), cosine_diffusion_schedule)
# loaded_model.load_state_dict(model_state_dict)


train_diffusion(diffusion_model, 8, model_path=M_PATH, save_model=False)


# gen_images = loaded_model.generate(n_images=10, diffusion_steps=20, normalizer_path=NORMALIZER_PATH)
# display_images_torch(gen_images)
