from jax import numpy as jnp
from jax import jit
from jax import Array
from typing import Callable, List
from sources.common.constants import EPS


def linear_layer() -> Callable[[Array, Array, Array], Array]:
    """
    Calculates the output of the function given the input, weights, and biases.
    Parameters:
        x (Array): The input array.
        W (Array): The weight array.
        b (Array): The bias array.
    Returns:
        y (Array): The output array.
    """

    @jit
    def f(x: Array, W: Array, b: Array):
        """
        Calculates the output of the function given the input, weights, and biases.

        Parameters:
            x (Array): The input array.
            W (Array): The weight array.
            b (Array): The bias array.

        Returns:
            y (Array): The output array.
        """
        y = jnp.dot(x, W) + b[jnp.newaxis, :]
        return y

    return f


def activation(typ: str, **kwargs) -> Callable[[Array], Array]:
    """
    Generate the activation function based on the given type.

    Parameters:
        typ (str): The type of activation function to generate. Valid options are 'sigmoid', 'tanh', 'relu'.
        **kwargs: Additional keyword arguments.

    Returns:
        Callable[[Array], Array]: The generated activation function.

    Raises:
        None.
    """
    if typ == "sigmoid":

        def f(y):
            z = jnp.exp(y)
            return z / (1 + z)

    elif typ == "tanh":

        def f(y):
            z = jnp.tanh(y)
            return z

    elif typ == "relu":

        def f(y):
            if "alpha" in kwargs.keys():
                alpha = kwargs["alpha"]
            else:
                alpha = 0.0
            z = (y > 0) * y + alpha * (y < 0) * y
            return z

    else:
        print("Warning, safe mode activated")

        def f(y):
            return y

    return f


def dense_layer(typ: str, **kwargs) -> Callable[[Array, Array, Array], Array]:
    """
    Returns a callable function that applies a dense layer operation.

    Parameters:
    - typ (str): The type of activation function to be applied.
    - **kwargs: Additional keyword arguments to be passed to the activation function.

    Returns:
    - Callable[[Array, Array, Array], Array]: A callable function that takes in three arrays (x, W, b) and returns an array.

    Example:
    dense_layer('relu')(x, W, b) returns the result of applying a dense layer operation with ReLU activation to the input arrays x, W, b.
    """

    @jit
    def f(x: Array, W: Array, b: Array) -> Array:
        """
        Applies a linear layer followed by an activation function to the input data.

        Args:
            x (Array): The input data array.
            W (Array): The weight array for the linear layer.
            b (Array): The bias array for the linear layer.

        Returns:
            Array: The output array after applying the linear layer followed by the activation function.
        """
        y = linear_layer()(x, W, b)
        return activation(typ, **kwargs)(y)

    @jit
    def g(x: Array, Ws: List[Array]) -> Array:
        """
        A function that takes an array `x` and a list of arrays `Ws` as input and returns the result of calling the function `f` with `x`, `Ws[0]`, and `Ws[1]` as arguments.

        Parameters:
            x (Array): The input array.
            Ws (List[Array]): The list of arrays.

        Returns:
            Array: The result of calling the function `f` with the given arguments.
        """
        return f(x, Ws[0], Ws[1])

    return g


def batch_norm(
    typ: str,
) -> Callable[[Array, Array, Array, Array, Array], Array]:
    @jit
    def f(
        x: Array, mu: Array, sigma: Array, alpha: Array, beta: Array
    ) -> Array:
        """
        This function takes in five arrays: `x`, `mu`, `sigma`, `alpha`, and `beta`. It performs element-wise operations on these arrays to calculate the value of `y` and returns the result.

        Parameters:
            - x (Array): The input array.
            - mu (Array): Thex mean array.
            - sigma (Array):x The standard deviation array.
            - alpha (Array): The alpha array.
            - beta (Array): The beta array.

        Returns:
            - Array: The calculated result array `y`.
        """
        xhat = (x - mu[jnp.newaxis, ...]) / sigma[jnp.newaxis, ...]
        y = (xhat + alpha[jnp.newaxis, ...]) * beta[jnp.newaxis, ...]
        return y

    if typ == "inference":

        @jit
        def g(x: Array, Ws: List[Array]) -> Array:
            """
            This function takes in an array `x` and a list of arrays `Ws` as parameters. It then passes `x` and the elements of `Ws` to the function `f` and returns the result. The function is decorated with `@jit`, which indicates that it is a just-in-time compiled function. The function has the following signature:

            Parameters:
            - x (Array): The input array.
            - Ws (List[Array]): The list of arrays.

            Returns:
            - Array: The result of calling the function `f` with the input arguments.
            """
            return f(x, Ws[0], Ws[1], Ws[2], Ws[3])

    else:

        @jit
        def g(x: Array, mu: Array, sigma: Array, Ws: List[Array]) -> Array:
            return f(x, mu, sigma, Ws[2], Ws[3])

    return g


@jit
def compute_lnorm(z: Array, p: int, lo: float) -> float:
    """
    Compute the L-norm of an array.

    Parameters:
        z (Array): The input array.
        p (int): The power of the norm.
        lo(float): The scaling factor.

    Returns:
        float: The computed L-norm.
    """
    return jnp.sum(lo * (jnp.abs(z) ** p))


def loss(typ: str, quantile: float = 0.5) -> Callable[[Array, Array], Array]:
    if typ == "mse":

        @jit
        def compute_loss(ypred: Array, y: Array) -> float:
            """
            Calculate the mean squared error between predicted values and actual values.

            Parameters:
            - ypred (Array): An array of predicted values.
            - y (Array): An array of actual values.

            Returns:
            - float: The mean squared error between predicted values and actual values.
            """
            return jnp.mean((y - ypred) ** 2)

    elif typ == "median":

        @jit
        def compute_loss(ypred: Array, y: Array) -> float:
            """
            Calculate the mean absolute error between predicted values and actual values.

            Parameters:
            - ypred (Array): An array of predicted values.
            - y (Array): An array of actual values.

            Returns:
            - float: The mean squared error between predicted values and actual values.
            """
            return jnp.mean(jnp.abs(y - ypred))

    elif typ == "median":

        @jit
        def compute_loss(ypred: Array, y: Array) -> float:
            """
            Calculate the quantile predicted values and actual values.

            Parameters:
            - ypred (Array): An array of predicted values.
            - y (Array): An array of actual values.

            Returns:
            - float: The mean squared error between predicted values and actual values.
            """
            return jnp.mean(((y >= ypred) - quantile) * (y - ypred))

    elif typ == "cross_entropy":

        @jit
        def compute_loss(ypred: Array, y: Array) -> float:
            """
            Calculate the cross-entropy predicted values and actual values.

            Parameters:
            - ypred (Array): An array of predicted values.
            - y (Array): An array of actual values.

            Returns:
            - float: The mean squared error between predicted values and actual values.
            """
            return -jnp.mean(y * jnp.log(ypred))

    else:
        raise Exception("Not implemented")
    return compute_loss


@jit
def mask(mask: Array, z: Array) -> Array:
    return jnp.array([mask * z, (1.0 - mask) * z])


def optimizer(
    typ: str,
) -> Callable[[Array, Array, Array, float, Array, Array, Array], List[Array]]:
    """
    Returns a callable function that computes the step for different types of optimization algorithms based on the given metadata.

    Parameters:
    - typ: A string that determines the type of algorithm.

    Returns:
    - A callable function that computes the step based on the given parameters.

    The returned function takes the following parameters:
    - w: An array representing the weights.
    - ms: An array representing the momentum values.
    - g: An array representing the gradients.
    - lr: A float representing the learning rate.
    - betas: An array representing the beta values.
    - lps: An array representing the penalization l1 and l2, only ok for the simple sgd and if not included in the model itself.
    - lds: An array representing the weight decays.

    Returns:
    - A list of arrays representing the updated weights and momentum values.
    """

    if typ == "sgd_with_momentum":

        @jit
        def compute_step(
            w: Array,
            ms: Array,
            g: Array,
            lr: float,
            betas: Array,
            lps: Array,
            lds: Array,
            clip: Array = 1.0,
        ) -> Array:
            m, beta1 = ms[0], betas[0]
            lp1, lp2 = lps[0], lps[1]
            ld1, ld2 = lds[0], lds[1]
            penalized_g = (
                jnp.clip(g, -clip, clip) + lp1 * jnp.sign(w) + lp2 * w
            )
            weight_decay = ld1 * jnp.sign(w) + ld2 * w
            new_m1 = (beta1 * m + (1.0 - beta1) * penalized_g) / (1.0 - beta1)
            new_m = jnp.array([new_m1])
            new_w = w - lr * (new_m1 + weight_decay)
            return jnp.concatenate([jnp.array([new_w]), new_m], 0)

    elif typ == "adam_w":

        @jit
        def compute_step(
            w: Array,
            ms: Array,
            g: Array,
            lr: float,
            betas: Array,
            lps: Array,
            lds: Array,
            clip: Array = 1.0,
        ) -> Array:
            m1, m2, beta1, beta2 = ms[0], ms[1], betas[0], betas[1]
            lp1, lp2 = lps[0], lps[1]
            ld1, ld2 = lds[0], lds[1]
            penalized_g = (
                jnp.clip(g, -clip, clip) + lp1 * jnp.sign(w) + lp2 * w
            )
            weight_decay = ld1 * jnp.sign(w) + ld2 * w
            new_m1 = (beta1 * m1 + (1.0 - beta1) * penalized_g) / (1.0 - beta1)
            new_m2 = (beta2 * m2 + (1.0 - beta2) * penalized_g**2) / (
                1.0 - beta2
            )
            new_m = jnp.array([new_m1, new_m2])
            new_w = w - lr * (new_m1 / (new_m2 + EPS) ** 0.5 + weight_decay)
            return jnp.concatenate([jnp.array([new_w]), new_m], 0)

    elif typ == "batch_norm":

        @jit
        def compute_step(
            w: Array,
            ms: Array,
            g: Array,
            lr: float,
            betas: Array,
            lps: Array,
            lds: Array,
        ) -> Array:
            new_w = w - lr * g
            return jnp.concatenate([jnp.array([new_w])], 0)

    else:

        @jit
        def compute_step(
            w: Array,
            ms: Array,
            g: Array,
            lr: float,
            betas: Array,
            lps: Array,
            lds: Array,
            clip: Array = 1.0,
        ) -> Array:
            lp1, lp2 = lps[0], lps[1]
            ld1, ld2 = lds[0], lds[1]
            penalized_g = (
                jnp.clip(g, -clip, clip) + lp2 * w + lp1 * jnp.sign(w)
            )
            weight_decay = ld1 * jnp.sign(w) + ld2 * w
            new_w = w - lr * (penalized_g + weight_decay)
            return jnp.concatenate([jnp.array([new_w])], 0)

    return compute_step
