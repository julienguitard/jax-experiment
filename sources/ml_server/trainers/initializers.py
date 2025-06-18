import json
import re
import copy
import numpy as np
from jax import config
from jax import numpy as jnp
from jax import random
from jax import jit
from jax import Array
from typing import Callable
from typing import List
from typing import Tuple
from sources.common.collections import lmap
from sources.common.collections import lmapzip as lmz
from sources.ml_server.models_.jax_decorators import vmap_2ary_decorator


def initialize_state(metadata, key):
    def initialize_weight(
        type: str, shape: Tuple, ind_2: int
    ) -> Callable[[], Array]:
        if type == "batch_norm":

            @jit
            def initialize_weight_() -> Array:
                return jnp.ones(shape) * (ind_2 in [1, 3])

        else:

            @jit
            def initialize_weight_() -> Array:
                return 1.0 * random.normal(key, shape)

        return initialize_weight_

    def initialize_weights() -> List[List[Array]]:
        weights = lmap(
            lambda x: lmap(
                lambda y: initialize_weight(
                    x["type"], y["weight_shape"], y["ind_2"]
                )(),
                x["weights"],
            ),
            metadata["layers"],
        )

        return weights

    @jit
    def initialize_weight_moments(vec: Array, weight: Array) -> Array:
        return vmap_2ary_decorator(lambda x, y: x * y)(vec, weight)

    @jit
    def initialize_moments(
        weights: List[List[Array]],
    ) -> List[List[Array]]:
        moments = lmz(
            lambda m, w: lmz(
                lambda m_, w_: initialize_weight_moments(
                    m_["optimizer"]["b_momentum"], w_
                ),
                zip(m["weights"], w),
            ),
            zip(metadata["layers"], weights),
        )

        return moments

    return initialize_weights, initialize_moments
