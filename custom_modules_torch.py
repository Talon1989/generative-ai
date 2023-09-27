import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# For rgb images in_channels = 3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.width = width
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=self.width, kernel_size=1)
        # affine determines if there are learnable parameters
        self.b_n = nn.BatchNorm2d(num_features=in_channels, affine=False)
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=self.width, kernel_size=3, padding=1)
        self.swish = Swish()
        self.conv_2 = nn.Conv2d(in_channels=self.width,
                                out_channels=self.width, kernel_size=3, padding=1)

    def forward(self, x):
        inp_width = x.shape[1]  # number of channels
        if inp_width == self.width:
            residual = x
        else:
            # increase number of channels
            residual = self.conv_1x1(x)
        x = self.b_n(x)
        x = self.swish(self.conv_1(x))
        x = self.conv_2(x)
        return x + residual


class DownBlock(nn.Module):  # in_channels are lists
    def __init__(self, in_channels, width, block_depth):
        super().__init__()
        self.block_depth = block_depth
        # self.residual_blocks = [ResidualBlock(in_channels, width) for _ in range(block_depth)]
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(in_channels[i], width) for i in range(block_depth)]
        )
        self.down_sample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x, skips = x
        # skips = skips_.copy()
        counter = 1
        for block in self.residual_blocks:
            x = block(x)
            skips.append(x)
            counter += 1
        x = self.down_sample(x)
        return [x, skips]


class UpBlock(nn.Module):  # in_channels are lists
    def __init__(self, in_channels, width, block_depth):
        super().__init__()
        self.block_depth = block_depth
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.residual_blocks = [ResidualBlock(in_channels, width) for _ in range(block_depth)]
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(in_channels[i], width) for i in range(block_depth)]
        )

    def forward(self, x):
        x, skips = x
        x = self.up_sample(x)
        for block in self.residual_blocks:
            x = torch.cat([x, skips.pop()], dim=1)  # second dimension is channels in torch (first is batch)
            x = block(x)
        return x, skips
