import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utilities import Swish


'''
creating a regression model using pytorch and tqdm
'''

#  https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
house_data = pd.read_csv('data/king-county-house-data.csv')
columns_to_check = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'
    ]
columns_to_check_2 = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront',
        'condition', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'
    ]
columns_to_normalize = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
columns_to_normalize_2 = ['sqft_living', 'sqft_basement', 'yr_built', 'yr_renovated']
MODEL_PATH = '/home/fabio/PycharmProjects/generative-ai/data/models/house-regressor/dict.pth'


def data_analysis():

    # figuring out if we need to use date as a feature
    dates = house_data['date'].to_numpy()
    # dates = house_data[:, 1].to_numpy()
    years = np.array([int(d[0:4]) for d in dates])
    print('oldest year sale: %d | newest year sale: %d' % (min(years), max(years)))
    # no need to use dates since years as so close, feature will most likely be irrelevant

    # check how many 'waterfront' houses there are consider them in the features
    # or remove them as outliers
    waterfront_count = defaultdict(int)
    for element in house_data['waterfront']:
        waterfront_count[element] += 1
    print('ratio of houses with waterfront: %d/%d' % (waterfront_count[1], waterfront_count[0]))
    # there are enough to consider using them as feature


def clean_data():
    # filtered_data = house_data.dropna(subset=columns_to_check)
    # filtered_data = filtered_data[columns_to_check]
    filtered_data = house_data[columns_to_check].dropna()
    return filtered_data


data = clean_data()

y = data['price'].to_numpy().reshape([-1, 1])
X_data_frame = data.drop('price', axis=1)

indices_to_normalize = X_data_frame.columns.get_indexer(columns_to_normalize)
X = X_data_frame.to_numpy()
X[:, -1] = LabelEncoder().fit_transform(X[:, -1])  # create label encoding for zips
# normalizing selected columns
X[:, indices_to_normalize] = StandardScaler().fit_transform(X[:, indices_to_normalize])


# train_size = int(X.shape[0] * (3/4))
# train_dataset = TensorDataset(
#     torch.from_numpy(X[0:train_size]).float(), torch.from_numpy(y[0: train_size]).float()
# )
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataset = TensorDataset(torch.from_numpy(X[train_size: X.shape[0]]).float(),
#                              torch.from_numpy(y[train_size: X.shape[0]]).float())
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def prep_iris_dataloader():
    iris = pd.read_csv('data/iris.csv')
    dataset = TensorDataset(
        torch.tensor(iris.iloc[:, 0:-2].to_numpy(), dtype=torch.float64),
        torch.tensor(iris.iloc[:, -2].to_numpy(), dtype=torch.float64))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 2**5).double(),
        #     nn.BatchNorm1d(2 ** 5).double(),
        #     Swish(),
        #     nn.Dropout(p=1/2),
        #     nn.Linear(2**5, 2**6).double(),
        #     nn.BatchNorm1d(2 ** 6).double(),
        #     Swish(),
        #     nn.Dropout(p=1/2),
        #     nn.Linear(2**6, 1).double()
        # )
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 2**4).double(),
        #     # nn.BatchNorm1d(2**4).double(),
        #     nn.ReLU(),
        #     nn.Dropout(p=1/2),
        #     nn.Linear(2**4, 2**4).double(),
        #     # nn.BatchNorm1d(2**4).double(),
        #     nn.ReLU(),
        #     nn.Linear(2**4, 2**5).double(),
        #     # nn.BatchNorm1d(2**5).double(),
        #     nn.ReLU(),
        #     nn.Linear(2**5, 2**4).double(),
        #     # nn.BatchNorm1d(2**4).double(),
        #     nn.ReLU(),
        #     nn.Linear(2**4, 1).double()
        # )
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2**4),
            # nn.BatchNorm1d(2**4).double(),
            nn.ReLU(),
            nn.Linear(2**4, 2**4),
            # nn.BatchNorm1d(2**4).double(),
            nn.ReLU(),
            nn.Linear(2**4, 2**5),
            # nn.BatchNorm1d(2**5).double(),
            nn.ReLU(),
            nn.Linear(2**5, 2**4),
            # nn.BatchNorm1d(2**4).double(),
            nn.ReLU(),
            nn.Linear(2**4, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


# WORKING BETTER THAN OG
class OtherRegressor(nn.Module):
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


regressor = Regressor(input_dim=X.shape[1])
# iris_regressor = Regressor(input_dim=3)


def train_model(model: nn.Module,  t_dataloader: DataLoader, v_dataloader: DataLoader,
                lr: float = 1/5_000, n_epochs=2_000, save_model=True):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    lowest_loss = np.inf
    with tqdm(total=n_epochs) as pbar:
        for e in range(1, n_epochs+1):
            # training
            train_losses = []
            for x, y in t_dataloader:
                model.train()
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().numpy())
            # # validation
            # val_losses = []
            # model.eval()
            # for x, y in v_dataloader:
            #     predictions = model(x)
            #     loss = criterion(predictions, y.reshape([-1, 1]))
            #     val_losses.append(loss.detach().numpy())
            train_losses_mean = np.mean(train_losses)
            # val_losses_mean = np.mean(val_losses)
            pbar.set_description('Epoch %d | Train Loss: %.3f'
                                 % (e, train_losses_mean))
            pbar.update(1)
            scheduler.step()
            if save_model and e % 100 == 0 and train_losses_mean < lowest_loss:
                print('\nSaving model ...')
                torch.save(
                    model.state_dict(),
                    MODEL_PATH)
                print('Model saved.')
                lowest_loss = train_losses_mean


# regressor.load_state_dict(torch.load(MODEL_PATH))
# train_model(regressor, train_dataloader, test_dataloader, n_epochs=20_000, save_model=False)
train_model(regressor, dataloader, dataloader, n_epochs=5_000, save_model=True)


x_batch, y_batch = next(iter(dataloader))
print(regressor(x_batch[0:3]))
print()
print(y_batch[0:3])


# train_model(iris_regressor, prep_iris_dataloader(), None, n_epochs=500, lr=1/1_000, save_model=False)
# x, y = next(iter(prep_iris_dataloader()))
# preds = iris_regressor(x).detach()
# scores = r2_score(y, preds)
# print('R2 Scores: %.5f' % scores)

#  loss: 56_273_107_012









































































































































































































































































