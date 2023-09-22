import numpy as np
import tensorflow as tf
keras = tf.keras


"""
utility class with callback methods
"""


# KERAS


class SaveModel(keras.callbacks.Callback):

    def __init__(self, model, path):
        super().__init__()
        self.model = model
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        # self.model.save_weights(filepath='%s-callback.h5' % self.path)
        self.model.save(filepath=self.path)


class DiffusionSaveModel(keras.callbacks.Callback):

    def __init__(self, model, path):
        super().__init__()
        self.model = model
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.model.network.save(filepath=self.path)














