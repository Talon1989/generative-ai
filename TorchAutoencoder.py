import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


LATENT_SPACE_DIM = 32
# EN_OUT_DIM = 2048
EN_OUT_SHAPE = torch.tensor((128, 4, 4))


# mnist_dataset = datasets.MNIST(
#     root='../data', train=True, download=True, transform=transforms.ToTensor()
# )
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
mnist_dataset = datasets.MNIST(
    root='../', train=True, download=True, transform=transform
)
mnist_dataloader = DataLoader(dataset=mnist_dataset, shuffle=True, batch_size=64)
#  access to unmodified (with transforms) data, not representative of dataset
torch_data = mnist_dataset.data.unsqueeze(dim=1)


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # padding=0 is valid, padding=1 is same
#         self.layer_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1)
#         self.layer_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
#         self.layer_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
#         self.flatten = nn.Flatten()
#         self.encoder_output = nn.Linear(2048, LATENT_SPACE_DIM)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.layer_1(x))
#         x = self.relu(self.layer_2(x))
#         x = self.relu(self.layer_3(x))
#         shape_before_flattening = x.shape[1:]  # (128, 4, 4), np.prod([128,4,4]) = 2048
#         x = self.flatten(x)
#         x = self.encoder_output(x)
#         return x, shape_before_flattening


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        en_out_dim = int(torch.prod(EN_OUT_SHAPE))
        # padding=0 is valid, padding=1 is same
        self.layer_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.layer_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.layer_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.encoder_output = nn.Linear(en_out_dim, LATENT_SPACE_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.flatten(x)
        x = self.encoder_output(x)
        return x


encoder = Encoder()

images, labels = next(iter(mnist_dataloader))
# _, s_b_f = encoder(images)
latent_space = encoder(images)


class Decoder(nn.Module):
    def __init__(self):

        class Reshape(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(x.size(0), *self.shape)

        super().__init__()
        en_out_dim = int(torch.prod(EN_OUT_SHAPE))
        self.dense = nn.Linear(LATENT_SPACE_DIM, en_out_dim)
        self.reshape = Reshape(shape=EN_OUT_SHAPE)
        # self.layer_1 = nn.ConvTranspose2d()

    def forward(self, x):
        x = self.dense(x)
        print(x.shape)
        x = self.reshape(x)
        print(x.shape)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


decoder = Decoder()

































































































































































