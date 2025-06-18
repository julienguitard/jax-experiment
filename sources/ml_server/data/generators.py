from jax import numpy as jnp
from jax import jit
from jax import random
from typing import Tuple, List

from data.transformers import add_axes


def data_generator(
    key: jnp.ndarray, n_samples: int, n0: int, n3: int, degree: int = 4
) -> jnp.ndarray:
    """
    Generates data using the given key, number of samples, number of features, degree, and number of output features.
    Parameters:
        key (numpy.ndarray): The key used for generating random data.
        n_samples (int): The number of data samples to generate.
        n0 (int): The number of features in the input data.
        n3 (int): The number of output features.
        degree (int, optional): The degree of the polynomial features. Defaults to 4.
    Returns:
        numpy.ndarray: The generated data with shape (n_samples, n0 + n3).
    """

    @jit
    def generate_data() -> jnp.ndarray:
        # Generate random data
        x = random.normal(key, (n_samples, n0))

        # Generate monomials
        monoms = [
            x[:, i, jnp.newaxis] ** jnp.arange(0, degree)[jnp.newaxis, :]
            for i in range(0, n0)
        ]
        monoms_ = [
            jnp.swapaxes(add_axes(m, n0), 1, i + 1)
            for (i, m) in enumerate(monoms)
        ]
        monoms__ = monoms_[0]

        # Multiply monomials together
        for m in monoms_[1:]:
            monoms__ = monoms__ * m

        # Generate coefficients
        coefs = 0.1 * random.normal(
            key,
            [
                (i == 0)
                + (i > 0) * (i < (n0 + 1)) * degree
                + (i == (n0 + 1)) * n3
                for i in range(0, n0 + 2)
            ],
        )

        # Multiply coefficients with monomials
        y_ = coefs * monoms__

        # Sum the y_ values
        for m in monoms_:
            y_ = jnp.sum(y_, -2)

        # Concatenate x and y_
        xy = jnp.concatenate([x, y_], 1)

        return xy

    return generate_data
