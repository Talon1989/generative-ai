import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F


TRAIN_RATIO = 7/10
ALPHA = 1/1_000
EPOCHS = 500


class IrisDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature_sample = torch.tensor(self.features[idx])
        target_sample = torch.tensor(self.targets[idx])
        return feature_sample, target_sample


class CustomNN(nn.Module):
    """
    classification dense nn
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_shape: np.array):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for h in hidden_shape:
            self.hidden_layers.append(nn.Linear(in_dim, h, dtype=torch.float64))
            in_dim = h
        # defining a new object for output layer since it has different activation
        self.output_layer = nn.Linear(in_dim, out_dim, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x


class CustomNNHardcoded(nn.Module):
    """
    classification dense nn
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer_1 = nn.Linear(in_dim, 16, dtype=torch.float64)
        self.layer_2 = nn.Linear(16, 16, dtype=torch.float64)
        self.layer_3 = nn.Linear(16, 32, dtype=torch.float64)
        self.layer_4 = nn.Linear(32, 32, dtype=torch.float64)
        self.output_layer = nn.Linear(32, out_dim, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def one_hot_transformation(a: np.array) -> np.array:
    """
    :param a: label encoded 1D np.array
    :return:
    """
    assert len(a.shape) == 1
    n_unique = len(np.unique(a))
    one_hot = np.zeros(shape=[a.shape[0], n_unique])
    for idx, val in enumerate(a):
        one_hot[idx, int(val)] = 1
    return one_hot


iris = pd.read_csv('data/iris.csv')
X, y = iris.iloc[:, 0:-1].to_numpy(), iris.iloc[:, -1].to_numpy()
y = LabelEncoder().fit_transform(y)
y = one_hot_transformation(y)


iris_dataset = IrisDataset(X, y)
train_size = int(TRAIN_RATIO * len(iris_dataset))
test_size = len(iris_dataset) - train_size
train_dataset, val_dataset = random_split(iris_dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)


model = CustomNN(in_dim=X.shape[1], out_dim=y.shape[1], hidden_shape=np.array([16, 16, 32]))
# model = CustomNNHardcoded(in_dim=X.shape[1], out_dim=y.shape[1])


optimizer = torch.optim.Adam(params=model.parameters(), lr=ALPHA)
criterion = nn.MSELoss()


def train_model():
    model.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, torch.tensor(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('EP %d | Loss: %.3f' % (ep, total_loss))
    model.eval()























































































































































































































