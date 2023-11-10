import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utilities import one_hot_transformation


iris = pd.read_csv('data/iris.csv')


# CLASSIFICATOR


# x_matrix, y_vector = iris.iloc[:, 0:-1].to_numpy(), iris['variety'].to_numpy()
# y_vector = LabelEncoder().fit_transform(y_vector)
# y_matrix = one_hot_transformation(y_vector)
# dataset = TensorDataset(
#     torch.tensor(x_matrix, dtype=torch.float64),
#     torch.tensor(y_matrix, dtype=torch.float64))
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
#
# class Classificator(nn.Module):
#     def __init__(self, input_dim, output_dim, dtype=torch.float64):
#         super().__init__()
#         torch.set_default_dtype(dtype)
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 2**3),
#             nn.ReLU(),
#             nn.Linear(2**3, 2**4),
#             nn.ReLU(),
#             nn.Linear(2**4, 2**5),
#             nn.ReLU(),
#             nn.Linear(2**5, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         output = self.layers(x)
#         return output
#
#
# x_batch, y_batch = next(iter(dataloader))
# classificator = Classificator(input_dim=4, output_dim=3)
# optimizer = torch.optim.Adam(params=classificator.parameters(), lr=1/1_000)
# criterion = nn.MSELoss()
#
#
# for epoch in range(1, 1001):
#     losses = []
#     for x, y in dataloader:
#         optimizer.zero_grad()
#         classificator.train()
#         preds = classificator(x)
#         loss = criterion(preds, y)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.detach().numpy())
#     if epoch % 10 == 0:
#         print('Epoch %d | Loss: %.4f' % (epoch, np.sum(losses)))
#
#
# print('Results of the training')
# preds = np.argmax(classificator(x_batch).detach(), axis=1)
# y_s = np.argmax(y_batch, axis=1)
# print('Accuracy percentage: %.2f%%' % (torch.sum(preds == y_s).numpy() / len(preds) * 100))


# REGRESSOR


x_matrix, y_vector = iris.iloc[:, 0:-2].to_numpy(), iris.iloc[:, -2].to_numpy()
dataset = TensorDataset(
    torch.tensor(x_matrix, dtype=torch.float64),
    torch.tensor(y_vector.reshape([-1, 1]), dtype=torch.float64))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


class Regressor(nn.Module):
    def __init__(self, input_dim, dtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(dtype)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2**3),
            nn.ReLU(),
            nn.Linear(2**3, 2**4),
            nn.ReLU(),
            nn.Linear(2**4, 2**5),
            nn.ReLU(),
            nn.Linear(2**5, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


regressor = Regressor(3)
optimizer = torch.optim.Adam(params=regressor.parameters(), lr=1/1_000)
criterion = nn.MSELoss()


# todo train the model



















































































































































































