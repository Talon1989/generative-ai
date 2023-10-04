import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader


iris = pd.read_csv('data/iris.csv')
X, y = iris.iloc[:, 0:3].to_numpy(), iris.iloc[:, 3].to_numpy()
y = np.reshape(y, [-1, 1])


dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True
)
