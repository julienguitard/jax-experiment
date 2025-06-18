from jax import numpy as jnp
from jax import jit
from jax import random
from jax import Array
from typing import Any, Callable, List

from sources.common.decorators import logger_decorator


def column_splitter(index: int) -> Callable[[Array], List[Array]]:
    """
    Split a 2D array into two columns.

    Parameters:
        z (Array): The input 2D array to be split.

    Returns:
        List[Array]: A list containing two arrays. The first array contains the columns of `z` from the beginning up to `index`. The second array contains the columns of `z` from `index` onwards.
    """

    @jit
    def column_split(z: Array) -> List[Array]:
        """
        Split a 2D array into two columns.

        Parameters:
            z (Array): The input 2D array to be split.

        Returns:
            List[Array]: A list containing two arrays. The first array contains the columns of `z` from the beginning up to `index`. The second array contains the columns of `z` from `index` onwards.
        """
        return [z[:, :index], z[:, index:]]

    return column_split


def random_sorter(key: float) -> Callable[[Array], Array]:
    @jit
    def random_sort(z):
        """
        Randomly sorts the elements of a given array along the first axis.
        Parameters:
            z (ndarray): The input array to be sorted.
        Returns:
            ndarray: The sorted array.
        """
        u = random.normal(key, [z.shape[0]])
        s = jnp.argsort(u)
        z_ = z[s, :]
        return z_

    return random_sort


@jit
def compute_var_(x: Array) -> Array:
    """
    Compute the variance of the given array.
    Parameters:
    - x (Array): The input array.
    Returns:
    - Array: The variance of the input array as a matrix.
    """
    x_ = x - jnp.mean(x, 0)[jnp.newaxis, :]
    one_ = 0.0 * x_[:, 0] + 1.0
    # jax compute variance ignore the classical correction to N-1
    return jnp.matmul(jnp.transpose(x_), x) / (jnp.sum(one_) - 0.0)


@jit
def compute_var(x: Array) -> Array:
    """
    This function computes the variance of the given array.
    Args:
        x (Array): The input array.
    Returns:
        Array: The variance of the input array as a vector.
    """
    x_ = x - jnp.mean(x, 0)[jnp.newaxis, :]
    one_ = 0.0 * x_[:, 0] + 1.0
    return jnp.diag(compute_var_(x))


@jit
def normalize(x: Array) -> Array:
    """
    Normalize the input array by subtracting the mean and dividing by the standard deviation columnwise.
    Parameters:
    - x (Array): The input array to be normalized.
    Returns:
    - normalized_x (Array): The normalized array obtained after applying the normalization process.
    """
    m = jnp.mean(x, 0)[jnp.newaxis, :]
    v = compute_var(x)[jnp.newaxis, :]
    normalized_x = (x - m) / jnp.sqrt(v)
    return normalized_x


def batchifier(n: int) -> Callable[[Array], List[Array]]:
    """
    This function takes in an array `z` and returns a list of arrays. The function utilizes the `jit` decorator from the `numba` library to optimize performance.

    Parameters:
        - z (Array): The input array.

    Returns:
        - List[Array]: A list of arrays obtained by splitting the input array `z` into batches of size `n`.
    """

    @jit
    def batchify(z: Array) -> List[Array]:
        """
        This function takes in an array `z` and returns a list of arrays. The function utilizes the `jit` decorator from the `numba` library to optimize performance.

        Parameters:
            - z (Array): The input array.

        Returns:
            - List[Array]: A list of arrays obtained by splitting the input array `z` into batches of size `n`.
        """
        n_s = z.shape[0]
        return [z[(n * i) : (n * (i + 1)), :] for i in range(0, n_s // n)]

    return batchify


def dispatcher(typ: str, index: int = None) -> Callable[[Array], List[Array]]:
    """
    A function that performs tensor dispatch.
    Parameters:
        typ (str): The type of dispatch to be performed.
        index (int, optional): The index to be used in the dispatch. Defaults to None.
    Returns:
        Callable[[Array], List[Array]]: A function that performs the tensor dispatch.
    Raises:
        Exception: If the dispatch type is not implemented.
    """
    if typ == "column_splitter":

        @jit
        def tensor_dispatch(b):
            """
            A function that performs tensor dispatch.

            Parameters:
                b (Tensor): The input tensor.

            Returns:
                dict: A dictionary containing two keys 'x' and 'y', with corresponding values obtained from column_splitter(index)(b).

            """
            return dict(zip(["x", "y"], column_splitter(index)(b)))

    else:
        raise Exception("Not implemented")
    return tensor_dispatch


def add_axes(x: Array, n: int) -> Array:
    """
    Generates a new array by adding axes to an existing array.

    Parameters:
        x (Array): The input array.
        n (int): The number of axes to add.

    Returns:
        Array: The resulting array with the added axes.
    """
    if n == 0:
        y = jnp.array(x)
    elif n > 0:
        y_ = add_axes(x, n - 1)
        y = y_[..., jnp.newaxis]
    else:
        raise Exception("Not implemented")
    return y
