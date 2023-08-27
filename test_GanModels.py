import unittest
import tensorflow as tf
import numpy as np
from unittest import TestCase
from GanModels import Discriminator, Generator


class TestGanModels(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = Generator()
        cls.discriminator = Discriminator()

    def test_generator_output(self):
        latent_dim = 100
        batch_size = 32
        latent_vector = tf.random.normal(shape=[batch_size, latent_dim], mean=0., stddev=1.)
        output = self.generator(latent_vector)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 64)
        self.assertEqual(output.shape[2], 64)

    def test_discriminator_output(self):
        def are_values_between(arr, min_val, max_val):
            return all(min_val <= val <= max_val for val in arr)
        batch_size = 8
        images = np.load(file='data/test_train_8_batch.npy')
        output = self.discriminator(images)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)
        self.assertTrue(are_values_between(output, 0, 1))




























