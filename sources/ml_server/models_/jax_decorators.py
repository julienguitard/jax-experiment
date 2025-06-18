from typing import Callable
from typing import Any
from jax import numpy as jnp
from jax import Array
from jax import grad
from jax import vmap
from sources.common.constants import EPS
from data.transformers import compute_var
from sources.common.decorators import logger_decorator


def batch_norm_composition_decorator(
    typ: str,
) -> Callable[[Callable], Callable]:
    def inner(func: Callable) -> Callable:
        if typ != "batch_norm":

            def gunc(x, *args, **kwargs):
                y = func(x, *args, **kwargs)
                return [
                    y,
                    jnp.mean(y, 0),
                    (compute_var(y) + EPS) ** 0.5,
                ]

        else:

            def gunc(x, *args, **kwargs):
                return [func(x, *args, **kwargs)]

        return gunc

    return inner


def grad_decorator(func):
    return grad(func, argnums=1)


def vmap_2ary_decorator(func):
    return vmap(func, (0, None), 0)


def batch_norm_differentiate_decorator(il, typ):
    def inner(func):
        if typ == "batch_norm":

            def gunc(x, ws, y):
                return func(x, ws)[il][1]

        else:

            @grad_decorator
            def gunc(x, ws, y):
                return func(x, ws, y)

        return gunc

    return inner
