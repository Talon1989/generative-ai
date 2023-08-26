import time

import numpy as np
import pandas
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
keras = tf.keras


def onehot_transformation(y):
    Y = np.zeros([len(np.unique(y)), y.shape[0]])
    for idx, value in enumerate(y):
        Y[int(value), idx] = 1
    return Y.T


iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, :-1].to_numpy()
y_ = iris.iloc[:, -1].to_numpy()
y_ = LabelEncoder().fit_transform(y_)
Y_ = onehot_transformation(y_)


model = keras.models.Sequential([
    keras.layers.Dense(units=2**4, activation='relu', input_shape=(4,)),
    keras.layers.Dense(units=2**5, activation='relu'),
    keras.layers.Dense(units=2**6, activation='relu'),
    keras.layers.Dense(units=3, activation='sigmoid')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1/1_000),
    # loss=keras.losses.MeanSquaredError(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)


#  custom callback to slow down the fit method such that I can read checkpoint callback
class SlowdownCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        time.sleep(1)


PATH = 'data/z_checkpoints_model_checkpoint.ckpt'
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=PATH, save_weights_only=True, monitor='val_loss', save_best_only=True, verbose=1
)
X_train, X_test, y_train, y_test = train_test_split(X_, Y_, train_size=3/4, shuffle=True)
model.fit(X_train, y_train, epochs=500, validation_data=[X_test, y_test], callbacks=[checkpoint_callback, SlowdownCallback()])


















































































































