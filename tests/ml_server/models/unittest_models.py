import unittest
from jax import numpy as jnp
from typing import Callable
from sources.ml_server.data.transformers import compute_var
from deprecated.models_.models import (
    layer_generator,
    batch_norm_composition_wrapper,
)
from sources.ml_server.models_.formulas import dense_layer, batch_norm


class TestModels(unittest.TestCase):
    def setUp(self):
        self.wrapper_func = batch_norm_composition_wrapper("batch_norm")

    def test_layer_generator_batch_norm(self):
        # Generate mock functions
        metadata = {"type": "batch_norm"}
        layer = layer_generator()(metadata)
        # Generate expected function
        expected_layer = batch_norm()
        # Mock data to compare the two functions executions
        x = jnp.array([[1, 2, 3], [1, 2, 5], [-1, 2, -1], [-1, 0, 5]])
        mu = jnp.mean(x, 0)
        sigma = compute_var(x) ** 0.5
        alpha = jnp.array([0, 0, 0])
        beta = jnp.array([1, 1, 1])
        output = layer(x, [mu, sigma, alpha, beta])
        expected_result = expected_layer(x, [mu, sigma, alpha, beta])
        print(output, expected_result)

        self.assertTrue(jnp.array_equal(output, expected_result))

    def test_layer_generator_dense_layer(self):
        metadata = {"type": "dense_layer"}
        layer = layer_generator()(metadata)
        expected_layer = dense_layer("relu")
        self.assertIsInstance(layer, Callable)

    def test_layer_generator_unknown_type(self):
        metadata = {"type": "unknown_type"}
        layer = layer_generator()(metadata)
        self.assertIsInstance(layer, Callable)

    def test_batch_norm_type(self):
        result = self.wrapper_func(jnp.array([1, 2, 3]))
        self.assertTrue(jnp.array_equal(result[0], jnp.array([1, 2, 3])))
        self.assertTrue(jnp.array_equal(result[1], jnp.zeros(3)))
        self.assertTrue(jnp.array_equal(result[2], jnp.array([1, 2, 3])))

    def test_other_type(self):
        wrapper_func = batch_norm_composition_wrapper("other_type")
        result = self.wrapper_func(jnp.array([1, 2, 3]))
        self.assertTrue(jnp.array_equal(result[0], jnp.array([1, 2, 3])))
        self.assertTrue(
            jnp.array_equal(result[1], jnp.mean(jnp.array([1, 2, 3]), 0))
        )
        self.assertTrue(
            jnp.array_equal(result[2], compute_var(jnp.array([1, 2, 3])))
        )


if __name__ == "__main__":
    unittest.main()
