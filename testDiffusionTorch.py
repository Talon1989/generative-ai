import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from TorchDiffusion import UNET, DiffusionModel, train_data, cosine_diffusion_schedule


class TestDiffusionTorch(TestCase):


    @classmethod
    def setUpClass(cls) -> None:
        print('setting up class')
        cls.unet = UNET(3)
        cls.noises, _ = cosine_diffusion_schedule(
            torch.ones(64, 1, 1, 1, device=torch.device('cpu')) - 1 / 10 * 2
        )
        cls.images, _ = next(iter(train_data))  # batch_size of images
        cls.normalizer_path = '/home/talon/PycharmProjects/generative-ai/data/normalizer_data'

    def test_image_normalization(self):
        self.assertEqual(self.images.shape, torch.Size([64, 3, 64, 64]))
        rgb_mean = self.images.mean(dim=[0, 2, 3])  # not on 1 because it's rbg
        rgb_std = self.images.std(dim=[0, 2, 3])  # not on 1 because it's rbg
        normalize_transform = transforms.Normalize(rgb_mean, rgb_std)
        normalized_images = normalize_transform(self.images)
        # plt.imshow(images[0].permute(1, 2, 0))
        # plt.show()
        # plt.imshow(normalized_images[0].permute(1, 2, 0))
        # plt.show()
        # plt.clf()

    def test_diffusion_times(self):
        def cosine_diffusion_schedule(diffusion_times):
            if type(diffusion_times) is not torch.Tensor:
                diffusion_times = torch.tensor(diffusion_times)
            noise_rates = torch.sin(diffusion_times * torch.pi / 2)
            signal_rates = torch.cos(diffusion_times * torch.pi / 2)
            return noise_rates, signal_rates
        batch_size = self.images.shape[0]
        diffusion_times = torch.rand(size=(batch_size, 1, 1, 1))
        noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times)
        noises = torch.randn_like(self.images)
        noisy_images = (signal_rates * self.images) + (noise_rates * noises)

    def test_normalize(self):
        normalizer = DiffusionModel.get_normalizer(self.images)
        means, stds = normalizer.mean, normalizer.std
        batch_size = self.images.shape[0]
        new_images = means.view(1, 3, 1, 1) + self.images * stds.view(1, 3, 1, 1)
        # new_means, new_stds = DiffusionModel.update_stats(  # it's not static anymore
        #     prev_mean=means, prev_std=stds, prev_count=batch_size, images=self.images
        # )
        # print(new_means)
        # print(new_means.view(1, 3, 1, 1))
        # plt.imshow(new_images[0].permute(1, 2, 0))
        # plt.show()
        # plt.clf()

    def test_current_images(self):
        diffusion_times = torch.ones(size=(64, 1, 1, 1)) - 1/10 * 3/7
        noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times)
        t = (noise_rates * self.images)

    def test_normalize_data(self):
        with open(self.normalizer_path, 'r') as f:
            lines = f.readlines()
            means = torch.tensor(list(map(float, lines[0].split())))
            stds = torch.tensor(list(map(float, lines[1].split())))






















