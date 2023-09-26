import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from DiffusionTorch import UNET, a, images


class TestDiffusionTorch(TestCase):


    @classmethod
    def setUpClass(cls) -> None:
        print('setting up class')
        cls.unet = UNET(3)
        cls.noises = a  # batch_size of noises
        cls.images = images  # batch_size of images

    def test_image_normalization(self):
        self.assertEqual(self.images.shape, torch.Size([64, 3, 64, 64]))
        rgb_mean = images.mean(dim=[0, 2, 3])  # not on 1 because it's rbg
        rgb_std = images.std(dim=[0, 2, 3])  # not on 1 because it's rbg
        normalize_transform = transforms.Normalize(rgb_mean, rgb_std)
        normalized_images = normalize_transform(images)
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
        batch_size = images.shape[0]
        diffusion_times = torch.rand(size=(batch_size, 1, 1, 1))
        noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times)
        noises = torch.randn_like(images)
        noisy_images = (signal_rates * images) + (noise_rates * noises)




















