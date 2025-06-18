import unittest
from jax import numpy as jnp
from jax import random
from data.transformers import (
    column_splitter,
    random_sorter,
    compute_var,
    normalize,
    batchifier,
    dispatcher,
    add_axes,
)


class TestTransformers(unittest.TestCase):
    def setUp(self):
        self.key = random.PRNGKey(0)
        self.x = jnp.array(
            [[i * 10 + j for j in range(0, 10)] for i in range(0, 100)]
        )

    def test_column_splitter(self):
        splitter = column_splitter(3)
        result = splitter(self.x)
        self.assertEqual(len(result), 2)
        self.assertTrue(
            jnp.array_equal(
                result[0],
                jnp.array(
                    [[i * 10 + j for j in range(0, 3)] for i in range(0, 100)]
                ),
            )
        )
        self.assertTrue(
            jnp.array_equal(
                result[1],
                jnp.array(
                    [[i * 10 + j for j in range(3, 10)] for i in range(0, 100)]
                ),
            )
        )

    def test_random_sorter(self):
        sorter = random_sorter(self.key)
        result = sorter(self.x)
        self.assertEqual(result.shape, self.x.shape)
        self.assertNotEqual(jnp.all(result == self.x), True)

    def test_compute_var(self):
        result = compute_var(self.x)
        self.assertEqual(result.shape, (10,))
        self.assertTrue(jnp.array_equal(result, jnp.var(self.x, axis=0)))

    def test_normalize(self):
        result = normalize(self.x)
        self.assertEqual(result.shape, self.x.shape)
        self.assertAlmostEqual(
            0.1 * jnp.mean(jnp.abs(jnp.mean(result, axis=0))), 0.0
        )
        self.assertAlmostEqual(
            jnp.mean(jnp.abs(jnp.std(result, axis=0) - 1.0)), 0.0
        )
        self.assertAlmostEqual

    def test_batchifier(self):
        batchify = batchifier(2)
        result = batchify(self.x)
        self.assertEqual(len(result), 50)
        self.assertTrue(jnp.array_equal(result[0], self.x[0:2, :]))

    def test_dispatcher(self):
        dispatcher_func = dispatcher("column_splitter", index=4)
        result = dispatcher_func(self.x)
        self.assertEqual(len(result), 2)
        self.assertTrue(
            jnp.array_equal(
                result["x"],
                jnp.array(
                    [[i * 10 + j for j in range(0, 4)] for i in range(0, 100)]
                ),
            )
        )
        self.assertTrue(
            jnp.array_equal(
                result["y"],
                jnp.array(
                    [[i * 10 + j for j in range(4, 10)] for i in range(0, 100)]
                ),
            )
        )

    def test_add_axes(self):
        # Test case 1: Adding 0 axes
        x1 = jnp.array([1, 2, 3])
        n1 = 0
        expected_output1 = jnp.array([1, 2, 3])
        self.assertTrue(jnp.array_equal(add_axes(x1, n1), expected_output1))

        # Test case 2: Adding 1 axis
        x2 = jnp.array([[1, 2, 3], [4, 5, 6]])
        n2 = 1
        expected_output2 = jnp.array([[[1], [2], [3]], [[4], [5], [6]]])
        self.assertTrue(jnp.array_equal(add_axes(x2, n2), expected_output2))

        # Test case 3: Adding 2 axes
        x3 = jnp.array([1, 2, 3])
        n3 = 2
        expected_output3 = jnp.array([[[1]], [[2]], [[3]]])
        self.assertTrue(jnp.array_equal(add_axes(x3, n3), expected_output3))

        # Test case 4: Adding negative number of axes
        x4 = jnp.array([[1, 2, 3], [4, 5, 6]])
        n4 = -1
        with self.assertRaises(Exception):
            add_axes(x4, n4)

        # Test case 5: Adding non-integer number of axes
        x5 = jnp.array([1, 2, 3])
        n5 = 1.5
        with self.assertRaises(Exception):
            add_axes(x5, n5)


if __name__ == "__main__":
    unittest.main()
