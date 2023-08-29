import unittest
import tensorflow as tf
import numpy as np
from unittest import TestCase
from GanModels import Discriminator, Generator, CGANGenerator, CGANDiscriminator


class TestGanModels(TestCase):

    @classmethod
    def setUpClass(cls):
        g_latent_dim = np.random.randint(1, 200)
        g_label_dim = np.random.randint(1, 20)
        cls.generator = Generator(z_dim=(g_latent_dim, ))
        cls.discriminator = Discriminator()
        cls.cgan_generator = CGANGenerator(z_dim=(g_latent_dim, ), label_dim=(g_label_dim, ))
        cls.cgan_discriminator = CGANDiscriminator(image_size=(64, 64, 1, ), label_size=(64, 64, 1, ))
        cls.g_latent_dim = g_latent_dim
        cls.g_label_dim = g_label_dim

    def test_generator_output(self):
        batch_size = 32
        latent_vector = tf.random.normal(shape=[batch_size, self.g_latent_dim], mean=0., stddev=1.)
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

    def test_cgan_generator_output(self):
        batch_size = 32
        latent_vector = tf.random.normal(shape=[batch_size, self.g_latent_dim], mean=0., stddev=1.)
        latent_label = tf.random.normal(shape=[batch_size, self.g_label_dim], mean=0., stddev=1.)
        output = self.cgan_generator([latent_vector, latent_label])
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 64)
        self.assertEqual(output.shape[2], 64)

    def test_cgan_discriminator_output(self):
        batch_size = 8
        images = np.load(file='data/test_train_8_batch.npy')
        output = self.cgan_discriminator([images, images])
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)




























