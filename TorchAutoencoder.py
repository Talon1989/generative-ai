import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


LATENT_SPACE_DIM = 8
# EN_OUT_DIM = 2048
EN_OUT_SHAPE = torch.tensor((128, 4, 4))
MODEL_PATH = '/home/fabio/PycharmProjects/generative-ai/data/models/pytorch_autoencoder.pth'


# plt.plot([i**2 for i in range(10)])
# plt.show()


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
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1 / 1_000)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        z = self.encoder(x)
        reproduction = self.decoder(z)
        return reproduction

    def fit(self, n_epochs=500, save_model=False, show_generation=False):
        for ep in range(1, n_epochs):
            initial_time = time.time()
            autoencoder.train()
            losses = []
            for images, _ in mnist_dataloader:
                self.train()
                self.optimizer.zero_grad()
                predictions = self(images)
                loss = self.criterion(predictions, images)
                loss.backward()
                self.optimizer.step()
                loss_detached = loss.detach()
                losses.append(loss_detached)
                # print('loss: %.3f' % loss_detached)
            losses_mean, losses_std = np.mean(losses), np.std(losses)
            print('Episode %d | Losses mean: %.3f | Losses std: %.3f | Elapsed time: %.3f'
                  % (ep, losses_mean, losses_std, time.time() - initial_time))
            if show_generation:
                with torch.no_grad():
                    self.eval()
                    index = np.random.randint(0, len(mnist_dataset))
                    image, _ = mnist_dataset[index]
                    generation = self(torch.unsqueeze(image, dim=0))
                    generated_image = generation.squeeze().squeeze()
                    og_image = torch.squeeze(image)

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                    axes[0].imshow(og_image, cmap='gray')
                    axes[0].set_title('original image')
                    axes[0].axis('off')
                    axes[1].imshow(generated_image, cmap='gray')
                    axes[1].set_title('generated image')
                    axes[1].axis('off')

                    plt.show()
                    # plt.clf()
            if ep % 10 == 0 and save_model:
                print('\nSaving the model ...')
                torch.save(obj=self.state_dict(), f=MODEL_PATH)
                print('Model saved.\n')


print()
decoder = Decoder()
# latent_space = encoder(images)
# outputs = decoder(latent_space)
autoencoder = Autoencoder(encoder, decoder)


def run_training(n_epochs=500, save_model=False, show_generation=False):
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=1/1_000)
    criterion = nn.MSELoss()
    for ep in range(1, n_epochs):
        initial_time = time.time()
        autoencoder.train()
        losses = []
        for images, _ in mnist_dataloader:
            autoencoder.train()
            optimizer.zero_grad()
            predictions = autoencoder(images)
            loss = criterion(predictions, images)
            loss.backward()
            optimizer.step()
            loss_detached = loss.detach()
            losses.append(loss_detached)
            # print('loss: %.3f' % loss_detached)
        losses_mean, losses_std = np.mean(losses), np.std(losses)
        print('Episode %d | Losses mean: %.3f | Losses std: %.3f | Elapsed time: %.3f'
              % (ep, losses_mean, losses_std, time.time() - initial_time))
        if show_generation:
            with torch.no_grad():
                autoencoder.eval()
                index = np.random.randint(0, len(mnist_dataset))
                image, _ = mnist_dataset[index]
                generation = autoencoder(torch.unsqueeze(image, dim=0))
                generated_image = generation.squeeze().squeeze()
                og_image = torch.squeeze(image)

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                axes[0].imshow(og_image, cmap='gray')
                axes[0].set_title('original image')
                axes[0].axis('off')
                axes[1].imshow(generated_image, cmap='gray')
                axes[1].set_title('generated image')
                axes[1].axis('off')

                plt.show()
                # plt.clf()
        if ep % 10 == 0 and save_model:
            print('\nSaving the model ...')
            torch.save(obj=autoencoder.state_dict(), f=MODEL_PATH)
            print('Model saved.\n')

# outputs = autoencoder(images)
# output = outputs[50].detach().numpy().squeeze()
# plt.imshow(output, cmap='gray')
# plt.show()


# run_training(n_epochs=100, save_model=False, show_generation=False)
# autoencoder.fit()

# trained_autoencoder = Autoencoder(Encoder(), Decoder())
# trained_autoencoder.load_state_dict(torch.load(MODEL_PATH))
# with torch.no_grad():
#     trained_autoencoder.eval()
#     index = np.random.randint(0, len(mnist_dataset))
#     image, label = mnist_dataset[index]
#     generation = trained_autoencoder(torch.unsqueeze(image, dim=0))
#     generated_image = generation.squeeze().squeeze()
#     print(label)
#     plt.imshow(generated_image, cmap='gray')
#     plt.show()


class VariationalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        en_out_dim = int(torch.prod(EN_OUT_SHAPE))
        # padding=0 is valid, padding=1 is same
        self.layer_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.layer_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.layer_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.output_mean = nn.Linear(en_out_dim, LATENT_SPACE_DIM)
        self.output_log_var = nn.Linear(en_out_dim, LATENT_SPACE_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.flatten(x)
        z_mean = self.output_mean(x)
        z_log_var = self.output_log_var(x)
        return z_mean, z_log_var

    # def build_covariance_matrix(self, stds):
    #     n_batches, n_dim = stds.shape
    #     covariance_matrix = torch.zeros(size=[n_batches, n_dim, n_dim])
    #     for i in range(n_batches):
    #         covariance_matrix[i] = torch.diag(stds[i] ** 2)
    #     return covariance_matrix
    #
    # # no reparameterization trick
    # def sample_z(self, data, training=True):
    #     if training:
    #         self.train()
    #         means, stds = self(data)
    #     else:
    #         self.eval()
    #         with torch.no_grad():
    #             means, stds = self(data)
    #     distribution = torch.distributions.MultivariateNormal(
    #         means, self.build_covariance_matrix(stds))
    #     z = distribution.sample()
    #     return z
    #
    # def build_eye_covariance_matrix(self, stds):
    #     n_batches, n_dim = stds.shape
    #     covariance_matrix = torch.zeros(size=[n_batches, n_dim, n_dim])
    #     for i in range(n_batches):
    #         covariance_matrix[i] = torch.eye(n_dim)
    #     return covariance_matrix
    #
    # def sample_z_trick(self, data, training=True):
    #     if training:
    #         self.train()
    #         means, stds = self(data)
    #     else:
    #         self.eval()
    #         with torch.no_grad():
    #             means, stds = self(data)
    #     zeros = torch.zeros_like(means)
    #     distribution = torch.distributions.MultivariateNormal(
    #         zeros, self.build_eye_covariance_matrix(stds))
    #     sample = distribution.sample()
    #     z = means + torch.exp(stds) * sample
    #     return z


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module, reconstruction_weight=1.):
        """
        :param encoder: variational encoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1/1_000)

    def build_eye_covariance_matrix(self, stds):
        n_batches, n_dim = stds.shape
        covariance_matrix = torch.zeros(size=[n_batches, n_dim, n_dim])
        for i in range(n_batches):
            covariance_matrix[i] = torch.eye(n_dim)
        return covariance_matrix

    def sample_z_trick(self, data, training=True):
        if training:
            self.train()
            means, log_stds = self.encoder(data)
        else:
            self.eval()
            with torch.no_grad():
                means, log_stds = self.encoder(data)
        zeros = torch.zeros_like(means)
        distribution = torch.distributions.MultivariateNormal(
            zeros, self.build_eye_covariance_matrix(log_stds))
        sample = distribution.sample()
        z = means + torch.exp(log_stds) * sample
        return means, log_stds, z

    def forward(self, x):
        _, _, z = self.sample_z_trick(x)
        reproduction = self.decoder(z)
        return reproduction

    def fit(self, n_epochs=200, save_model=False, show_generation=False):
        for ep in range(1, n_epochs):
            initial_time = time.time()
            autoencoder.train()
            # reconstruction_losses, kl_losses, total_losses = [], [], []
            reconstruction_losses, kl_losses, total_losses = 0, 0, 0
            for images, _ in mnist_dataloader:
                self.train()
                self.optimizer.zero_grad()
                z_means, z_log_stds, z = self.sample_z_trick(images, training=True)
                reproduction = self.decoder(z)
                reconstruction_loss = self.reconstruction_weight \
                                      * self.reconstruction_criterion(reproduction, images)
                # kl_loss = - torch.sum(
                #     1/2 * (1 + 2 * z_log_stds - torch.square(z_means) - torch.exp(2 * z_log_stds)), dim=1
                # ).mean()
                z_log_vars = 2 * z_log_stds
                kl_loss = torch.sum(
                    -1/2 * (1 + z_log_vars - torch.square(z_means) - torch.exp(z_log_vars)), dim=1
                ).mean()
                total_loss = reconstruction_loss + kl_loss
                total_loss.backward()
                self.optimizer.step()
                # reconstruction_losses.append(reconstruction_loss.detach().numpy())
                # kl_losses.append(kl_loss.detach().numpy())
                # total_losses.append(total_loss.detach().numpy())
                reconstruction_losses += reconstruction_loss.detach().numpy()
                kl_losses += kl_loss.detach().numpy()
                # print(reconstruction_losses[-1])
                # print(kl_losses[-1])
            total_losses = reconstruction_losses + kl_losses
            # rec_mean, kl_mean = np.mean(reconstruction_losses), np.mean(kl_losses)
            # total_mean = np.mean(total_losses)
            print('Episode %d | Rec Losses: %.4f | KL Losses: %.5f | Total Losses: %.4f'
                  % (ep, reconstruction_losses, kl_losses, total_losses))
            if show_generation:
                with torch.no_grad():
                    self.eval()
                    index = np.random.randint(0, len(mnist_dataset))
                    image, _ = mnist_dataset[index]
                    generation = self(torch.unsqueeze(image, dim=0))
                    generated_image = generation.squeeze().squeeze()
                    og_image = torch.squeeze(image)

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                    axes[0].imshow(og_image, cmap='gray')
                    axes[0].set_title('original image')
                    axes[0].axis('off')
                    axes[1].imshow(generated_image, cmap='gray')
                    axes[1].set_title('generated image')
                    axes[1].axis('off')

                    plt.show()
                    # plt.clf()
            if ep % 10 == 0 and save_model:
                print('\nSaving the model ...')
                torch.save(obj=self.state_dict(), f=VAE_PATH)
                print('Model saved.\n')


v_encoder = VariationalEncoder()
decoder = Decoder()
vae = VariationalAutoEncoder(v_encoder, decoder)
# z_means, z_stds = v_encoder(images[0:3])
# z_cov_matrix = v_encoder.build_covariance_matrix(z_stds)
# mvn = torch.distributions.MultivariateNormal(z_means, z_cov_matrix)
# z = mvn.sample()
VAE_PATH = '/home/fabio/PycharmProjects/generative-ai/data/models/pytorch_vae.pth'
vae.fit(save_model=True)





































































































































