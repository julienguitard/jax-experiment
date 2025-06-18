import unittest
from jax import numpy as jnp
from sources.ml_server.data.transformers import normalize, compute_var
from sources.ml_server.models_.formulas import (
    linear_layer,
    activation,
    dense_layer,
    batch_norm,
    compute_lnorm,
    optimizer,
)


class TestFormulas(unittest.TestCase):
    def test_linear_layer(self):
        # Generate mock functions
        self.g = linear_layer()

        # Define the expected output
        x = jnp.array([[1, 2, 3], [1, 2, 5], [-1, 2, -1], [-1, 0, 5]])
        W = jnp.array([[1, 2, 3, 0], [4, 5, 6, -1], [7, 8, 9, -2]])
        b = jnp.array([[1, 2, -3, 4]])

        cases = [
            {
                "case": "shape",
                "result": linear_layer()(x, W, b).shape,
                "expected_output": [4, 4],
            },
        ]
        _ = cases.extend(
            [
                {
                    "case": "{}_{}".format(i, j),
                    "result": linear_layer()(
                        x,
                        jnp.array(
                            [
                                [
                                    (ii == i) * (jj == j) + 0.0
                                    for jj in range(0, 4)
                                ]
                                for ii in range(0, 3)
                            ]
                        ),
                        0 * b,
                    ),
                    "expected_output": x[..., i][..., jnp.newaxis]
                    * (jnp.arange(0, 4) == j),
                }
                for i in range(0, 3)
                for j in range(0, 4)
            ]
        )

        for case in cases:
            with self.subTest(case=case["case"]):
                self.assertTrue(
                    jnp.array_equal(case["result"], case["expected_output"])
                )

    def test_activation(self):
        # Generate mock functions
        self.fs = [activation(a) for a in ["sigmoid", "tanh", "relu"]]

        # Define the expected output
        x = jnp.array([[1, 2, 3], [1, 2, 5], [-1, 2, -1], [-1, 0, 5]])
        cases = [
            {
                "case": "sigmoid",
                "results": activation("sigmoid")(x),
                "expected_output": jnp.exp(x) / (1 + jnp.exp(x)),
            },
            {
                "case": "tanh",
                "results": self.fs[1](x),
                "expected_output": jnp.tanh(x),
            },
            {
                "case": "relu",
                "results": self.fs[2](x),
                "expected_output": jnp.maximum(0, x),
            },
        ]

        # Assert that the result matches the expected output
        for case in cases:
            with self.subTest(case=case["case"]):
                discrepancy = jnp.max(
                    jnp.abs(case["results"] - case["expected_output"])
                )
                self.assertAlmostEqual(discrepancy, 0.0)

    def test_dense_layer(self):
        # Generate mock functions
        self.g = dense_layer("relu")

        # Define the expected output
        x = jnp.array([[1, 2, 3], [1, 2, 5], [-1, 2, -1], [-1, 0, 5]])
        W = jnp.array([[1, 2, 3, 0], [4, 5, 6, -1], [7, 8, 9, -2]])
        b = jnp.array([[1, 2, -3, 4]])

        cases = [
            {
                "case": "shape",
                "result": self.g(x, [W, b]).shape,
                "expected_output": [4, 4],
            },
        ]
        _ = cases.extend(
            [
                {
                    "case": "{}_{}".format(i, j),
                    "result": self.g(
                        x,
                        [
                            jnp.array(
                                [
                                    [
                                        (ii == i) * (jj == j) + 0.0
                                        for jj in range(0, 4)
                                    ]
                                    for ii in range(0, 3)
                                ]
                            ),
                            0 * b,
                        ],
                    ),
                    "expected_output": x[..., i][..., jnp.newaxis]
                    * (x[..., i][..., jnp.newaxis] > 0)
                    * (jnp.arange(0, 4) == j),
                }
                for i in range(0, 3)
                for j in range(0, 4)
            ]
        )

        # Assert that the result matches the expected output
        for case in cases:
            with self.subTest(case=case["case"]):
                self.assertTrue(
                    jnp.array_equal(case["result"], case["expected_output"])
                )

    def test_batch_norm(self):
        # Generate mock functions
        self.g = batch_norm()

        # Define the expected output
        x = jnp.array([[1, 2, 3], [1, 2, 5], [-1, 2, -1], [-1, 0, 5]])
        mu = jnp.mean(x, 0)
        sigma = compute_var(x) ** 0.5
        alpha = jnp.array([0, 0, 0])
        beta = jnp.array([1, 1, 1])
        expected_output = normalize(x)
        results = self.g(x, [mu, sigma, alpha, beta])
        # Assert that the result matches the expected output
        self.assertTrue(jnp.array_equal(results, expected_output))

    def test_compute_lnorm(self):
        # Define the expected output
        z = jnp.array([1, 2, -3])
        p = 3
        li = 0.5
        expected_output = 18
        output = compute_lnorm(z, p, li)
        # Assert that the result matches the expected output
        self.assertEqual(output, expected_output)

    def test_optimizer(self):
        # Define the expected output
        metadata = {"type": "sgd_with_momentum"}
        w = jnp.array([1, 2, 3])
        ms = [jnp.array([0, 0, 0]), jnp.array([0.5, 0.5, 0.5])]
        g = jnp.array([2.0, 2.0, 2.0])
        lr = 0.1
        betas = jnp.array([0.9, 0.8])
        lps = jnp.array([0.0, 0.0])
        lds = jnp.array([0.1, 0.1])
        m, beta1 = jnp.array([0, 0, 0]), 0.9
        ld1, ld2 = 0.1, 0.1
        penalized_g = g
        weight_decay = ld1 * jnp.sign(w) + ld2 * w
        new_m1 = (beta1 * m + (1.0 - beta1) * penalized_g) / (1.0 - beta1)
        new_m = jnp.array([new_m1])
        new_w = w - lr * (new_m1 + weight_decay)
        expected_output = jnp.concatenate([jnp.array([new_w]), new_m], 0)
        result = optimizer(metadata)(w, ms, g, lr, betas, lps, lds)
        # Assert that the result matches the expected output
        print(result, expected_output)
        self.assertTrue(jnp.array_equal(result, expected_output))


if __name__ == "__main__":
    unittest.main()
