import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # keeps the first shape the same (batch)
        return x.view(x.size(0), *self.shape)


LEGO_PATH = '/home/fabio/datasets/lego-brick-images/'
LATENT_DIMS = 64
BATCH_SIZE = 64


lego_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
lego_dataset = datasets.ImageFolder(
    root=LEGO_PATH,
    transform=lego_transform
)
lego_dataloader = DataLoader(lego_dataset, batch_size=BATCH_SIZE, shuffle=True)


mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
mnist_dataset = datasets.MNIST(
    root='../', train=True, download=True, transform=mnist_transform
)
mnist_dataloader = DataLoader(dataset=mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)


# images, labels = next(iter(lego_dataloader))
# image = images[5].squeeze()
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()


# images, labels = next(iter(mnist_dataloader))
# image = images[5].squeeze()
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()


class Discriminator(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_channels, 64, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)
        self.conv_4 = nn.Conv2d(256, 1, kernel_size=(4, 4), stride=1, padding=0)
        self.batch_norm_2 = nn.BatchNorm2d(128, momentum=9/10)
        self.batch_norm_3 = nn.BatchNorm2d(256, momentum=9/10)
        self.swish = Swish()
        self.leaky_relu = nn.LeakyReLU(negative_slope=1/5)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=3/10)
        self.flatten = nn.Flatten()

    def forward(self, x:torch.Tensor):
        x = self.swish(self.conv_1(x))
        x = self.dropout(x)
        x = self.swish(self.conv_2(x))
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = self.swish(self.conv_3(x))
        x = self.batch_norm_3(x)
        x = self.dropout(x)
        x = self.sigmoid(self.conv_4(x))
        x = self.flatten(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dims, n_channels=1):
        super().__init__()
        self.latent_dims = latent_dims
        self.linear = nn.Linear(self.latent_dims, 256)
        self.deconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=(8, 8), stride=2, padding=1)
        self.deconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=(8, 8), stride=2, padding=1)
        self.deconv_3 = nn.ConvTranspose2d(64, n_channels, kernel_size=(4, 4), stride=2, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(128, momentum=9/10)
        self.batch_norm_2 = nn.BatchNorm2d(64, momentum=9/10)
        self.reshape = nn.Unflatten(1, (256, 1, 1))  # initial 1 is to keep the batch shape
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.reshape(x)
        x = self.swish(self.deconv_1(x))
        x = self.batch_norm_1(x)
        x = self.swish(self.deconv_2(x))
        x = self.batch_norm_2(x)
        x = self.sigmoid(self.deconv_3(x))
        return x


class GenerativeAdversarialNetwork:
    def __init__(self,discriminator:nn.Module, generator:nn.Module):
        self.discriminator = discriminator
        self.d_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=1/5_000)
        self.generator = generator
        self.g_optimizer = torch.optim.Adam(
            params=self.generator.parameters(), lr=1/5_000)
        self.criterion = nn.BCELoss()

    def fit(self, n_epochs=500, dataloader=mnist_dataloader, save_model=False):

        for ep in range(1, n_epochs+1):
            d_losses, g_losses = [], []

            for images, _ in dataloader:

                with torch.no_grad():
                    distribution = torch.distributions.Normal(0, 1)
                    latent_vectors = distribution.sample([images.shape[0], LATENT_DIMS])

                self.discriminator.train()
                self.generator.train()

                # computing discriminator gradients
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                d_real_preds = self.discriminator(images)
                real_labels = torch.ones_like(d_real_preds)
                with torch.no_grad():
                    generated_images = self.generator(latent_vectors)
                d_fake_preds = self.discriminator(generated_images)
                fake_labels = torch.zeros_like(d_fake_preds)
                d_real_loss = self.criterion(d_real_preds, real_labels)
                d_fake_loss = self.criterion(d_fake_preds, fake_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

                # computing generator gradients
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                generated_images = self.generator(latent_vectors)
                d_fake_preds = self.discriminator(generated_images)
                g_loss = self.criterion(d_fake_preds, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                d_losses.append(d_loss.detach().numpy())
                g_losses.append(g_loss.detach().numpy())

            d_loss_mean = float(np.mean(d_losses))
            g_loss_mean = float(np.mean(g_losses))
            print('Epoch %d | d loss: %.4f | g loss: %.4f' % (ep, d_loss_mean, g_loss_mean))


d = Discriminator(n_channels=1)
standard_gaussian = torch.distributions.Normal(0, 1)
latent_v = standard_gaussian.sample([32, LATENT_DIMS])
g = Generator(latent_dims=LATENT_DIMS)
# v = g(latent_v)
# gen_image = v[0].squeeze().detach().numpy()
# plt.imshow(gen_image, cmap='gray')
# plt.axis('off')
# plt.show()
gan = GenerativeAdversarialNetwork(d, g)
gan.fit()


































































































































































































































































































































































































































































































