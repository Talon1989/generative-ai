import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchmetrics
from custom_modules_torch import ResidualBlock, DownBlock, UpBlock


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


PATH = "/home/talon/datasets/flower-dataset/dataset"
EMA = 999 / 1_000
NOISE_EMBEDDING_SIZE = 32


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
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader


# shape is (64, 3, 64, 64): (batch_size, channels, dim_1, dim_2)
train_data = image_dataloader_from_directory(directory=PATH)
# print(train_data.dataset.classes)
# for imgs, lbs in train_data:
#     images, labels = imgs, lbs
#     print(images.shape)
#     break
images, labels = next(iter(train_data))
images = images.numpy()


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
    max_signate_rate = torch.tensor(19 / 20)
    start_angle = torch.acos(max_signate_rate)
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


IMAGE_SIZE = 64
BATCH_SIZE = 64
PATH = "/home/talon/datasets/flower-dataset/dataset"


class DiffusionModel(nn.Module):
    def __init__(self, unet_model: UNET, diff_schedule):
        super().__init__()
        self.network = unet_model
        self.ema_network = copy.deepcopy(self.network)
        self.diffusion_schedule = diff_schedule
        self.noise_loss_tracker = torchmetrics.MeanMetric()
        self.optimizer = torch.optim.AdamW(
            params=self.network.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()

    def get_normalizer(self, images):
        rgb_mean = images.mean(dim=[0, 2, 3])  # not on 1 because it's rbg
        rgb_std = images.std(dim=[0, 2, 3])  # not on 1 because it's rbg
        normalize_transform = torchvision.transforms.Normalize(rgb_mean, rgb_std)
        return normalize_transform

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
        pred_images = (noisy_images - (noise_rates * pred_noises)) / signal_rates
        return pred_noises, pred_images

    def train_step(self, images):

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
        self.optimizer.step()

        self.ema_soft_update()

        self.noise_loss_tracker.update(noise_loss)
        print('Noise loss : %.4f' % self.noise_loss_tracker.compute().item())

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        pass

    def denormalize(self, images):
        pass

    def generate(self, n_images, diffusion_steps):
        pass


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


def train_diffusion(n_epochs, model_path, save_model=False):
    total_batches = len(train_data)
    for epoch in range(n_epochs):
        start_time = time.time()
        for batch_number, (images, _) in enumerate(train_data, start=1):
            print('Epoch: %d | Batch %d/%d:' % (epoch + 1, batch_number, total_batches))
            diffusion_model.train_step(images)
        if save_model:
            torch.save(diffusion_model.state_dict(), f=model_path+'.pth')
            print('Model saved in %s' % (model_path+'.pth'))
        end_time = time.time()
        print('Elapsed time for training epoch %d : %.3f' % (epoch + 1, (start_time - end_time)/60))


M_PATH = '/home/talon/PycharmProjects/generative-ai/data/models/U-Net-Pytorch'


## LOAD TORCH MODEL
# model_state_dict = torch.load(M_PATH+'.pth')
# loaded_model = DiffusionModel(UNET(3), cosine_diffusion_schedule)
# loaded_model.load_state_dict(model_state_dict)


# train_diffusion(10, model_path=M_PATH, save_model=True)































































































































