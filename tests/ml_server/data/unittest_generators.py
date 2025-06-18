import unittest
from jax import numpy as jnp
from jax import random
from typing import Tuple, List

from data.generators import data_generator

# Define unit tests using the unittest module


class TestGenerators(unittest.TestCase):
    def test_data_generator_shape(self):
        key = random.PRNGKey(0)
        n_samples = 10
        n0 = 2
        n3 = 1
        degree = 4

        generator = data_generator(key, n_samples, n0, n3, degree)
        data = generator()

        expected_shape = (n_samples, n0 + n3)
        self.assertEqual(data.shape, expected_shape)


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
