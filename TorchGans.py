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
BATCH_SIZE = 32


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
        self.conv_1 = nn.Conv2d(n_channels, 64, kernel_size=(4, 4), stride=2, padding=1).double()
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1).double()
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1).double()
        self.conv_4 = nn.Conv2d(256, 1, kernel_size=(4, 4), stride=1, padding=0).double()
        self.batch_norm_2 = nn.BatchNorm2d(128, momentum=9/10).double()
        self.batch_norm_3 = nn.BatchNorm2d(256, momentum=9/10).double()
        self.swish = Swish()
        self.leaky_relu = nn.LeakyReLU(negative_slope=1/5)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=3/10)
        self.flatten = nn.Flatten()

    def forward(self, x:torch.Tensor):
        x = x.double()
        x = self.swish(self.conv_1(x))
        x = self.dropout(x)
        x = self.swish(self.conv_2(x))
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = self.swish(self.conv_3(x))
        x = self.batch_norm_3(x)
        x = self.dropout(x)
        x = self.sigmoid(self.conv_4(x))
        x = self.flatten(x).double()
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
                try:
                    d_fake_loss = self.criterion(d_fake_preds, fake_labels)
                except RuntimeError:
                    print('Runtime Error')
                    print('discriminator prediction of generated data:')
                    print(d_fake_preds.squeeze())
                    return
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


class WassersteinGenerativeAdversarialNetwork:
    def __init__(self,discriminator:nn.Module, generator:nn.Module,
                 l_constraint:float=1., d_steps:int=3, g_p_weight:float=10.):
        self.discriminator = discriminator
        self.d_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=1/5_000)
        self.generator = generator
        self.g_optimizer = torch.optim.Adam(
            params=self.generator.parameters(), lr=1/5_000)
        self.criterion = nn.BCELoss()
        self.l_constraint = l_constraint
        self.d_steps = d_steps
        self.g_p_weight = g_p_weight

    def gradient_penalty(self, real_data:torch.Tensor, fake_data:torch.Tensor):
        batch_size = real_data.shape[0]
        alpha = torch.randn(batch_size, 1, 1, 1)  # sampled from standard gaussian
        difference = fake_data - real_data
        interpolated_data = real_data + alpha * difference
        interpolated_data.requires_grad_(True)
        predictions = self.discriminator(interpolated_data)
        grads = torch.autograd.grad(
            outputs=predictions, inputs=interpolated_data,
            grad_outputs=torch.ones_like(predictions), create_graph=True
        )[0]
        norm = torch.sqrt(torch.sum(grads ** 2, dim=[1, 2, 3]))  # l2
        # avg square distance between l2 and l-constraint
        grad_penalty = torch.mean((norm - self.l_constraint) ** 2)
        return grad_penalty

    def fit(self, n_epochs=500, dataloader=mnist_dataloader, save_model=False):

        for ep in range(1, n_epochs+1):
            d_losses, g_losses = [], []

            for images, _ in dataloader:

                images = images.to(torch.float64)
                self.discriminator.train()
                self.generator.train()

                # computing discriminator gradients
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                d_w_loss = torch.tensor(0.)
                for _ in range(self.d_steps):
                    with torch.no_grad():
                        distribution = torch.distributions.Normal(0, 1)
                        latent_vectors = distribution.sample([images.shape[0], LATENT_DIMS])
                        generated_images = self.generator(latent_vectors)
                    d_real_preds = self.discriminator(images)
                    d_fake_preds = self.discriminator(generated_images)
                    if not torch.any(torch.isnan(d_fake_preds)):
                        wasserstein_loss = torch.mean(d_fake_preds - d_real_preds)
                        grad_penalty = self.gradient_penalty(images, generated_images)
                        d_loss = wasserstein_loss + self.g_p_weight * grad_penalty
                    else:
                        print('Discriminator real predictions')
                        print(d_real_preds)
                        print('Discriminator fake predictions')
                        print(d_fake_preds)
                        return
                    d_loss.backward()
                    self.d_optimizer.step()
                    d_w_loss += d_loss

                    # print(d_w_loss)

                # computing generator gradients
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                with torch.no_grad():
                    distribution = torch.distributions.Normal(0, 1)
                    latent_vectors = distribution.sample([images.shape[0], LATENT_DIMS])
                generated_images = self.generator(latent_vectors)
                d_fake_preds = self.discriminator(generated_images)
                g_loss = torch.mean(d_fake_preds)
                g_loss.backward()
                self.g_optimizer.step()

                # print(d_w_loss)
                # print(g_loss)

                d_losses.append(d_w_loss.detach().numpy())
                g_losses.append(g_loss.detach().numpy())

            # hardcoded path
            if ep % 2 == 0 and save_model:
                print('Saving generator model...')
                torch.save(obj=self.generator.state_dict(),
                           f='/home/fabio/PycharmProjects/generative-ai/data/models/pytorch_WGAN.pth')
                print('Model saved.')

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


# gan = GenerativeAdversarialNetwork(d, g)
# gan.fit()


wgan = WassersteinGenerativeAdversarialNetwork(d, g, l_constraint=1., d_steps=1)
# images, _ = next(iter(mnist_dataloader))
# images = images[0:3]
# distribution = torch.distributions.Normal(0, 1)
# latent_vectors = distribution.sample([images.shape[0], LATENT_DIMS])
# generations = wgan.generator(latent_vectors)
wgan.fit(save_model=True)

































































































































































































































































































































































































































































































