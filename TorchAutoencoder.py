import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


LATENT_SPACE_DIM = 2
# EN_OUT_DIM = 2048
EN_OUT_SHAPE = torch.tensor((128, 4, 4))


# mnist_dataset = datasets.MNIST(
#     root='../data', train=True, download=True, transform=transforms.ToTensor()
# )
transform = transforms.Compose([
    transforms.Resize((32, 32)),
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
        # print(x.shape)
        x = self.relu(self.layer_2(x))
        # print(x.shape)
        x = self.relu(self.layer_3(x))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.encoder_output(x)
        # print(x.shape)
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
                # keeps the first shape the same (batch)
                return x.view(x.size(0), *self.shape)

        super().__init__()
        en_out_dim = int(torch.prod(EN_OUT_SHAPE))
        self.dense = nn.Linear(LATENT_SPACE_DIM, en_out_dim)
        self.reshape = Reshape(shape=EN_OUT_SHAPE)
        # self.layer_1 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=2, padding=1)
        # self.layer_2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1)
        # self.layer_3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.layer_1 = nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.layer_2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)
        self.layer_3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1)
        self.decoder_output = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        # print(x.shape)
        x = self.reshape(x)
        # print(x.shape)
        x = self.relu(self.layer_1(x))
        # print(x.shape)
        x = self.relu(self.layer_2(x))
        # print(x.shape)
        x = self.relu(self.layer_3(x))
        # print(x.shape)
        x = self.sigmoid(self.decoder_output(x))
        # print(x.shape)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        reproduction = self.decoder(z)
        return reproduction


print()
decoder = Decoder()
# outputs = decoder(latent_space)
autoencoder = Autoencoder(encoder, decoder)
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=1/1_000)


'''
torch.Size([64, 32, 16, 16])
torch.Size([64, 64, 8, 8])
torch.Size([64, 128, 4, 4])
torch.Size([64, 2048])
torch.Size([64, 2])

torch.Size([64, 2048])
torch.Size([64, 128, 4, 4])
torch.Size([64, 128, 7, 7])
torch.Size([64, 64, 13, 13])
torch.Size([64, 32, 25, 25])
torch.Size([64, 1, 25, 25])
'''




























































































































































