import numpy as np
import torch
import torch.nn as nn
import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict


'''
creating a regression model using pytorch and tqdm
'''


house_data = pd.read_csv('data/king-county-house-data.csv')
columns_to_check = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'
    ]


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
    filtered_data = house_data.dropna(subset=columns_to_check)
    filtered_data = filtered_data[columns_to_check]
    return filtered_data


data = clean_data()
y = data['price'].to_numpy()
X = data.drop('price', axis=1).to_numpy()























































































































































































































































































