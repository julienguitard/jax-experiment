from jax import numpy as jnp
from jax import jit
from jax import grad
from jax import Array
from typing import Callable, List, Dict, Tuple, Any

from sources.common.collections import concatenate
from sources.common.constants import PERSISTENT_DATA_PATH
from sources.common.decorators import time_decorator
from sources.common.decorators import logger_decorator
from sources.common.decorators import nested_decorator
from sources.common.collections import concatenate
from sources.common.collections import lmap
from sources.common.collections import lreduce
from sources.common.collections import lzip
from sources.common.collections import lmapzip as lmz
from sources.common.collections import flatten
from sources.common.collections import flatten_tuple as tu
from sources.common.collections import lfilter
from sources.common.collections import ljoin
from sources.common.collections import enumerate_flat as ef
from sources.common.collections import dmap
from sources.common.collections import dfilter
from sources.common.collections import dicter
from sources.common.collections import merge
from sources.common.collections import rename
from sources.common.collections import select


from sources.ml_server.data.transformers import batchifier
from sources.ml_server.data.transformers import dispatcher
from sources.ml_server.data.transformers import compute_var
from sources.ml_server.models_.architectures import compute_weight_metadata
from sources.ml_server.models_.architectures import compute_layer_metadata
from sources.ml_server.models_.architectures import generate_architecture
from data.transformers import compute_var
from models_.formulas import batch_norm
from models_.formulas import dense_layer
from models_.formulas import compute_lnorm
from models_.formulas import optimizer

from sources.ml_server.models_.formulas import (
    dense_layer,
    batch_norm,
    compute_lnorm,
    optimizer,
)
from sources.ml_server.models_.jax_decorators import (
    batch_norm_composition_decorator,
    vmap_2ary_decorator,
)


def layer_generator() -> Callable[[Dict[str, Any]], Callable[[Array], Array]]:
    """
    Generates a layer generator function.

    Returns:
        Callable[[Dict[str, Any]], Callable[[Array], Array]]: A callable that takes a metadata dictionary as input and returns a callable that takes an array as input and returns an array.

    """

    def generate_layer(metadata: Dict[str, Any]) -> Callable[[Array], Array]:
        """
        Generates a layer based on the given metadata.

        Parameters:
            metadata (Dict[str, Any]): A dictionary containing the metadata for the layer.

        Returns:
            Callable[[Array], Array]: A callable that takes an input array and returns the output array.
        """
        if metadata["type"] == "batch_norm":
            layer = batch_norm()
        else:
            layer = dense_layer(metadata["type"])
        return layer

    return generate_layer


def loss_generator() -> (
    Callable[[Dict[str, Any]], Callable[[Array, Array], float]]
):
    def generate_loss(
        metadata: Dict[str, Any]
    ) -> Callable[[Array, Array], float]:
        """
        Generate a loss function based on the given metadata.

        Parameters:
        - metadata (Dict[str, Any]): A dictionary containing metadata for generating the loss function.

        Returns:
        - Callable[[Array, Array], float]: A function that calculates the mean squared error between predicted values and actual values.
        """
        if metadata["loss_type"] == "mse":

            @jit
            def lo(ypred: Array, y: Array) -> float:
                """
                Calculate the mean squared error between predicted values and actual values.

                Parameters:
                - ypred (Array): An array of predicted values.
                - y (Array): An array of actual values.

                Returns:
                - float: The mean squared error between predicted values and actual values.
                """
                return jnp.mean((y - ypred) ** 2)

        else:
            raise Exception("Not implemented")
        return lo

    return generate_loss


def select_weight_gradient_like(
    metadata: Dict[str, Any], grad_: Array, moments: List[Array]
) -> Array:
    if metadata["gradient_like"] == "gradient":
        res = grad_
    else:
        res = moments[metadata["ind_2"]]
    return res


def generate_model(metadata):
    layers_metadata = metadata["layers"]

    loss_ = loss_generator()(metadata)

    layers = lmap(
        lambda m: batch_norm_composition_decorator(m["type"])(
            jit(layer_generator()(m))
        ),
        layers_metadata,
    )

    @jit
    def model(x, ws):
        res = [[x, jnp.mean(x, 0), compute_var(x)]]
        for i_layer, layer in enumerate(layers):
            res = concatenate(res, [layer(res[-1][0], ws[i_layer])])
        return res[1:]

    @jit
    def loss(x, ws, y):
        return loss_(model(x, ws)[-1][0], y)

    def compute_lnorm_(
        weight_metadata: Dict[str, Any]
    ) -> Callable[[Array], float]:
        if "penalization" in weight_metadata.keys():

            def func(z: Array) -> float:
                return compute_lnorm(
                    z,
                    weight_metadata["penalization"]["penalization_type"],
                    weight_metadata["penalization"]["penalization_l"],
                )

        else:

            def func(z: Array) -> float:
                return 0.0

        return func

    penalization_functions = lmap(
        lambda x: lmap(lambda y: compute_lnorm_(y), x["weights"]),
        layers_metadata,
    )

    @jit
    def penalize(ws):
        return jnp.sum(
            jnp.array(
                flatten(
                    lmz(
                        lambda f, w: lmz(lambda f_, w_: f_(w_), zip(f, w)),
                        zip(penalization_functions, ws),
                    )
                )
            )
        )

    @jit
    def penalized_loss(x, ws, y):
        return loss(x, ws, y) + penalize(ws)

    penalized_loss_grad = grad(penalized_loss, argnums=1)

    def compute_grad_like(x, ws, y):
        grad_ = penalized_loss_grad(x, ws, y)
        moments = lmap(lambda x: x[1:], model(x, ws))
        nested_triples = lmz(
            lambda meta, gr, mo: lmap(
                tu,
                ljoin(
                    lambda a, b: True,
                    lzip(meta["weights"], gr),
                    [mo],
                ),
            ),
            zip(layers_metadata, grad_, moments),
        )

        return lmap(
            lambda li: lmz(select_weight_gradient_like, li),
            nested_triples,
        )

    return model, penalize, loss, penalized_loss, compute_grad_like


def generate_optimizer(
    metadata: Dict[str, Any]
) -> Callable[
    [List[List[Array]], List[List[Array]], List[List[Array]]],
    Tuple[List[List[Array]], List[List[Array]]],
]:
    optimizers = lmap(
        lambda la: lmap(
            lambda wm: lambda w, m, g: optimizer(
                wm["optimizer"]["optimizer_type"]
            )(
                w,
                m,
                g,
                wm["optimizer"]["learning_rate"],
                wm["optimizer"]["b_momentum"],
                wm["optimizer"]["lps"],
                wm["optimizer"]["lds"],
            ),
            la["weights"],
        ),
        metadata["layers"],
    )

    def optimize(
        ws: List[List[Array]], ms: List[List[Array]], gs: List[List[Array]]
    ) -> List[List[Array]]:
        return lmz(
            lambda i, ws_, ms_, gs_: lmz(
                lambda j, w, m, g: optimizers[i][j](w, m, g),
                ef(zip(ws_, ms_, gs_)),
            ),
            ef(zip(ws, ms, gs)),
        )

    def split_optimized(
        weights_moments: List[List[Array]],
    ) -> Tuple[List[List[Array]], List[List[Array]]]:
        return (
            lmap(lambda wm: lmap(lambda wm_: wm_[0], wm), weights_moments),
            lmap(lambda wm: lmap(lambda wm_: wm_[1:], wm), weights_moments),
        )

    @jit
    def optimize_split(
        ws: List[List[Array]], ms: List[List[Array]], gs: List[List[Array]]
    ) -> Tuple[List[List[Array]], List[List[Array]]]:
        return split_optimized(optimize(ws, ms, gs))

    return optimize_split
