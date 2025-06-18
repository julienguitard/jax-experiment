from jax import numpy as jnp
from jax import jit
from jax import grad
from jax import Array
from typing import Callable, List, Dict, Tuple, Any

from sources.common.collections import concatenate
from sources.common.constants import PERSISTENT_DATA_PATH
from sources.common.constants import EPS
from sources.common.decorators import time_decorator
from sources.common.decorators import logger_decorator
from sources.common.decorators import nested_decorator
from sources.common.decorators import nested_decorator_
from sources.common.collections import concatenate
from sources.common.collections import lmap
from sources.common.collections import lreduce
from sources.common.collections import lreduce_cumulatedly as lrc
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
from models_.formulas import loss
from models_.formulas import mask
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


def generate_layer(metadata: Dict[str, Any]) -> Callable[[Array], Array]:
    """
    Generates a layer based on the given metadata.

    Parameters:
        metadata (Dict[str, Any]): A dictionary containing the metadata for
        the layer.

    Returns:
        Callable[[Array], Array]: A callable that takes an input array and
        returns the output array.
    """
    if metadata["type"] == "batch_norm":
        layer = batch_norm("training")
    else:
        layer = dense_layer(metadata["type"])
    return layer


def generate_lnorm(
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


def generate_loss(metadata: Dict[str, Any]) -> Callable[[Array, Array], float]:
    """
    Generate a loss function based on the given metadata.

    Parameters:
    - metadata (Dict[str, Any]): A dictionary containing metadata for
    generating the loss function.

    Returns:
    - Callable[[Array, Array], float]: A function that calculates the loss
    from the predicted values and actual values.
    """

    if not ("loss_params" in metadata.keys()):
        metadata["loss_params"] = {}

    return loss(metadata["loss_type"], **metadata["loss_params"])


def grad_like_for_batch_norm(metadata: Dict[str, Any]) -> Callable:
    if metadata["type"] == "batch_norm":

        @jit
        def g(mu: Array, sigma: Array, Ws: List[Array]) -> List[Array]:
            return [Ws[0] - mu, Ws[1] - sigma, 0.0 * Ws[2], 0.0 * Ws[3]]

    else:

        @jit
        def g(Ws: List[Array]) -> List[Array]:
            return lmap(lambda w: 0.0 * w, Ws)

    return g


def generate_reducand(
    layer_: Callable,
    penalization_: Callable,
    grad_like_for_batch_norm_: Callable,
):
    @jit
    def reducand(previous_node: Tuple[Array], Ws: List[Array]) -> Tuple[Array]:
        proper_node = layer_(*previous_node[0], Ws)
        penalization_node = previous_node[1] + penalization_(Ws)
        grad_like_args_node = proper_node[1:]
        grad_like_node = grad_like_for_batch_norm_(*previous_node[2], Ws)

        return (
            proper_node,
            penalization_node,
            grad_like_args_node,
            grad_like_node,
        )

    return reducand


def generate_model(metadata):
    layers_metadata = metadata["layers"]

    loss = generate_loss(metadata)

    layers = lmap(
        lambda la: batch_norm_composition_decorator(la["type"])(
            jit(generate_layer(la))
        ),
        layers_metadata,
    )

    penalizations = lmap(
        lambda la: lmap(
            lambda y: lambda w: generate_lnorm(y)(w), la["weights"]
        ),
        layers_metadata,
    )

    summed_penalizations = lmap(
        lambda pen: lambda ws: jnp.sum(
            jnp.array(lmz(lambda iw, w: pen[iw](w), enumerate(ws)))
        ),
        penalizations,
    )

    grad_like_for_batch_norms = lmap(grad_like_for_batch_norm, layers_metadata)

    reducands = lmz(
        generate_reducand,
        zip(layers, summed_penalizations, grad_like_for_batch_norms),
    )

    @jit
    def model(x, ws):
        mu = jnp.mean(x, 0)
        sigma = (compute_var(x) + EPS) ** 0.5
        val = (
            [x, mu, sigma],
            0.0,
            [mu, sigma],
            [],
        )
        nodes = lrc(
            lambda val, ir: ir[1](val, ws[ir[0]]),
            val,
            enumerate(reducands),
        )
        return nodes

    @jit
    def penalized_loss(x, ws, y):
        print("model(x, ws)[-1][1].shape", model(x, ws)[-1][1].shape)
        return loss(model(x, ws)[-1][0][0], y) + model(x, ws)[-1][1]

    penalized_loss_grad = jit(grad(penalized_loss, argnums=1))

    @jit
    def compute_grad_like(x, ws, y):
        grad_ = penalized_loss_grad(x, ws, y)
        grad_like = lmap(lambda node: node[3], model(x, ws))
        return lmz(
            lambda g, gl: lmz(lambda g_, gl_: g_ + gl_, zip(g, gl)),
            zip(grad_, grad_like),
        )

    return model, loss, penalized_loss, penalized_loss_grad, compute_grad_like


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
