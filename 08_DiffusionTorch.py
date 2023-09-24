import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision


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
    diffusion_times = torch.tensor(diffusion_times)
    signal_rates = torch.cos(diffusion_times * torch.pi / 2)
    noise_rates = torch.sin(diffusion_times * torch.pi / 2)
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


linear = linear_diffusion_schedule(10)
