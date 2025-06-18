from jax import numpy as jnp
from jax import Array
from typing import Callable, List, Dict, Any

from sources.common.constants import PERSISTENT_DATA_PATH
from sources.common.decorators import time_decorator
from sources.common.decorators import logger_decorator
from sources.common.decorators import nested_decorator
from sources.common.collections import concatenate
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
from sources.common.collections import lmap
from sources.common.constants import EPS


def compute_weight_metadata(metadata, layer_metadata, index):
    res = {"ind_2": index}
    non_batch_norm_optimize_metadata = select(
        set(
            filter(
                lambda k: not ("batch_norm" in k),
                metadata["optimizer"].keys(),
            )
        )
    )(metadata["optimizer"])
    if (layer_metadata["type"] == "batch_norm") and (index < 2):
        res["weight_shape"] = (layer_metadata["shape"][0],)
        res["gradient_like"] = "moment"
        res["optimizer"] = {
            "optimizer_type": "batch_norm",
            "learning_rate": 1.0 - metadata["optimizer"]["b_batch_norm"],
            "b_momentum": jnp.array([]),
            "lps": jnp.zeros(2),
            "lds": jnp.zeros(2),
        }
    elif (layer_metadata["type"] == "batch_norm") and (index >= 2):
        res["weight_shape"] = (layer_metadata["shape"][0],)
        res["penalization"] = metadata["penalization"]
        res["gradient_like"] = "gradient"

        res["optimizer"] = non_batch_norm_optimize_metadata
    elif index == 0:
        res["weight_shape"] = layer_metadata["shape"]
        res["penalization"] = metadata["penalization"]
        res["gradient_like"] = "gradient"
        res["optimizer"] = non_batch_norm_optimize_metadata
    else:
        res["weight_shape"] = (layer_metadata["shape"][1],)
        res["penalization"] = metadata["penalization"]
        res["gradient_like"] = "gradient"
        res["optimizer"] = non_batch_norm_optimize_metadata
    return res


def compute_layer_metadata(metadata, compute_weight_metadata):
    l0 = lmap(
        dicter(["ind_0", "type", "size"]),
        ef(
            flatten(
                [
                    [("inputs", len(metadata["inputs"]))],
                    lmap(
                        lambda x: ("relu", x), metadata["hidden_layers_sizes"]
                    ),
                    [("linear", len(metadata["outputs"]))],
                ]
            )
        ),
    )
    l1 = lmap(
        merge,
        ljoin(
            lambda x, y: (x["ind_0"] + 1) == y["ind_0_"],
            l0,
            lmap(
                rename({"ind_0": "ind_0_", "type": "type_", "size": "size_"}),
                l0,
            ),
        ),
    )
    l2 = lmap(
        lambda x: {
            "ind_0": x["ind_0"],
            "type": x["type_"],
            "shape": (x["size"], x["size_"]),
        },
        l1,
    )

    l3 = lmap(
        lambda x: merge(({"ind_1": x[0]}, x[1])),
        ef(
            lmap(
                lambda x: merge((x[0], {"is_batch_norm": x[1]})),
                ljoin(
                    lambda x, y: True,
                    l2,
                    lfilter(
                        lambda y: (not y) or metadata["batch_normed"],
                        [True, False],
                    ),
                ),
            )
        ),
    )

    l4 = lmap(
        lambda x: merge(
            (
                select({"ind_0", "ind_1", "shape"})(x),
                {"type": (x["type"], "batch_norm")[x["is_batch_norm"]]},
            )
        ),
        l3,
    )

    l5 = lmap(
        lambda x: merge(
            (
                x,
                {
                    "weights": lfilter(
                        lambda y: y < (2 + 2 * (x["type"] == "batch_norm")),
                        range(0, 4),
                    )
                },
            )
        ),
        l4,
    )

    def foo(x):
        def goo(k, v):
            if k == "weights":
                res = (
                    k,
                    lmap(
                        lambda y: compute_weight_metadata(metadata, x, y),
                        v,
                    ),
                )
            else:
                res = (k, v)
            return res

        return goo

    l6 = lmap(lambda x: dmap(foo(x), x), l5)

    return l6


def generate_architecture(
    compute_weight_metadata, compute_layer_metadata, architecture_seed
):
    architecture = architecture_seed
    architecture["layers"] = compute_layer_metadata(
        architecture, compute_weight_metadata
    )

    return architecture
